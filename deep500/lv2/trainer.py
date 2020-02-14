import copy
import os
import numpy as np
import torch
import timeit
from typing import List, Optional, Type

from deep500.lv1.graph_executor import GraphExecutor
from deep500.lv2.summaries import TrainingStatistics, EpochSummary
from deep500.lv2.events.summary_generator import SummaryGeneratorEvent
from deep500.lv2.optimizer import Optimizer
from deep500.lv2.sampler import Sampler
from deep500.lv1.event import ExecutorEvent
from deep500.lv2.event import (TrainingEvent, OptimizerEvent, SamplerEvent, 
                               RunnerEvent, StopTraining)

from deep500.lv2.events import (SummaryGeneratorEvent, 
    SummaryGeneratorInferenceEvent, TerminalBarEvent)


def DefaultTrainerEvents(epochs):
    return [TerminalBarEvent(epochs)]


class Trainer(object):
    """ A training manager class that runs a training/test loop, with epochs 
        and invoking corresponding events. """
    def __init__(self,
                 training_sampler: Sampler,
                 validation_sampler: Optional[Sampler],
                 executor: GraphExecutor,
                 optimizer: Optimizer,
                 network_output: Optional[str] = None):
        """ Creates a Trainer object.
            @param training_sampler Sampler that samples training dataset.
            @param validation_sampler Sampler that samples validation dataset.
                                      Can be None.
            @param executor Graph executor to run the network.
            @param optimizer The optimizer to use for training.
            @param network_output The node name of the network prediction 
                                  (value or classification) for accuracy
                                  computation. Can be None.
        """
        self.train_set = training_sampler
        self.test_set = validation_sampler
        self.executor = executor
        self.optimizer = optimizer
        self.network_output = network_output

    def _train(self, stats, events, optimizer_events):
        self.train_set.reset()

        for event in events: event.before_training_set(self, stats, self.train_set)
        output = self.optimizer.train(len(self.train_set), self.train_set, 
                                      optimizer_events)
        for event in events: event.after_training_set(self, stats)

    def _test_accuracy(self, stats, events):
        if self.test_set is None: 
            return

        self.test_set.reset()

        for event in events: event.before_test_set(self, stats, self.test_set)
        for j, inp in enumerate(self.test_set):
            for event in events: event.before_test_batch(self, stats, inp)
            out = self.executor.inference(inp)
            for event in events: event.after_test_batch(self, stats, out)
        for event in events: event.after_test_set(self, stats)

    def run_loop(self, epochs,
                 events: List[TrainingEvent] = None,
                 collect_all_times: bool = False) -> TrainingStatistics:
        """
        Runs train and test set alternately for a given number of epochs.
        @param epochs Number of epochs to run the loop for.
        @param events A list of events to use in training/testing. 
                      Instances of RunnerEvent invoke the runner events,
                      instances of OptimizerEvent and SamplerEvent are also
                      invoked in the optimizer and sampler objects.
        @param collect_all_times Training statistics collect every latency
                                 of optimizer and executor steps.
        @return Training statistics for all epochs.
        """
        # Create statistics object
        stats = TrainingStatistics(self.train_set.batch_size, 
                                   (0 if self.test_set is None else 
                                        self.test_set.batch_size))
        # Set and distribute events
        if events is None:
            events = DefaultTrainerEvents(epochs)
        if collect_all_times:
            events.append(SummaryGeneratorInferenceEvent(stats))
        else:
            events.append(SummaryGeneratorEvent(stats))
        executor_events = [e for e in events if isinstance(e, ExecutorEvent)]
        optimizer_events = [e for e in events if isinstance(e, OptimizerEvent)]
        sampler_events = [e for e in events if isinstance(e, SamplerEvent)]
        events = [e for e in events if isinstance(e, RunnerEvent)]

        # Append events to executor and samplers
        self.executor.events.extend(executor_events)
        self.train_set.events.extend(sampler_events)
        if self.test_set is not None:
            self.test_set.events.extend(sampler_events)

        try:
            for event in events: event.before_training(self, stats)

            # Run test set prior to training
            self._test_accuracy(stats, events)

            for epoch in range(epochs):
                for event in events: event.before_epoch(epoch, self, stats)
                self._train(stats, events, optimizer_events)
                self._test_accuracy(stats, events)
                for event in events: event.after_epoch(epoch, self, stats)

        except (StopIteration, StopTraining):
            pass # If stopping was requested

        for event in events: event.after_training(self, stats)
              
        # Remove events from executor and samplers
        del self.executor.events[-len(executor_events):]
        del self.train_set.events[-len(sampler_events):]
        if self.test_set is not None:
            del self.test_set.events[-len(sampler_events):]

        return stats


class DCGanTrainer(Trainer):
    def __init__(self,
                 training_sampler: Sampler,
                 noise_sampler: Sampler,
                 D_executor: GraphExecutor,
                 G_executor: GraphExecutor,
                 D_optimizer: Optimizer,
                 G_optimizer: Optimizer,
                 D_input_node: str,
                 G_input_node: str,
                 network_output: Optional[str]=None,
                 validation_sampler: Optional[Sampler]=None):
        super().__init__(training_sampler, None, D_executor, D_optimizer, None)
        self.train_set = training_sampler
        self.noise_set = noise_sampler
        self.test_set = validation_sampler
        self.D_executor = D_executor
        self.G_executor = G_executor
        self.D_optimizer = D_optimizer
        self.G_optimizer = G_optimizer
        self.D_input_node = D_input_node
        self.G_input_node = G_input_node
        self.network_output = network_output
        self.fake_label = torch.ones(self.train_set.batch_size, 1, 1, 1).to(self.D_executor.devname)
        self.real_label = torch.zeros(self.train_set.batch_size, 1, 1, 1).to(self.D_executor.devname)

    def _train(self, stats, events, optimizer_events):
        self.train_set.reset()
        self.noise_set.reset()
        for event in events: event.before_training_set(self, stats, self.train_set)

        epoch_loss = EpochSummary(True, 0)
        for i in range(len(self.train_set)):
            images = self.train_set()
            noise = self.noise_set()

            for event in optimizer_events:
                event.before_optimizer_step(self.executor, self, images)
                start = timeit.timeit()
                loss_d, loss_g = self._train_algo_step(images, noise)
                stop = timeit.timeit()
                epoch_loss.wrong_batch.append((loss_d.item(), loss_g.item(), stop-start))


            for event in optimizer_events:
                event.after_optimizer_step(self.executor, self, loss_d, loss_g)

        stats.test_summaries.append(epoch_loss)

        for event in events: event.after_training_set(self, stats)

    def _train_algo_step(self, images, noise):
        # place to start optimizer_events

        self.D_executor.model.zero_grad()
        self.D_optimizer.op.zero_grad()
        # ------ train Discriminator ------
        # pass real samples
        self.D_executor.model._params[self.D_input_node] = torch.tensor(images[self.D_input_node]).to(
            self.D_executor.devname)
        self.D_executor.model._params['label'] = self.real_label
        loss_real = self.D_executor.model()
        loss_real.backward()

        # pass fake samples
        for name, val in noise.items():
            self.G_executor.model._params[name] = torch.tensor(val).to(self.G_executor.devname)

        fakes = self.G_executor.model()
        self.D_executor.model._params[self.D_input_node] = fakes.detach()
        self.D_executor.model._params['label'] = self.fake_label
        loss_fakes = self.D_executor.model()
        loss_fakes.backward()
        self.D_optimizer.op.step()
        loss_d = loss_fakes + loss_real

        # ----- train Generator -----
        self.G_executor.model.zero_grad()
        self.G_optimizer.op.zero_grad()

        self.D_executor.model._params[self.D_input_node] = fakes
        self.D_executor.model._params['label'] = self.real_label

        loss_g = self.D_executor.model()
        loss_g.backward()
        self.G_optimizer.op.step()

        return loss_d.detach().cpu().numpy(), loss_g.detach().cpu().numpy()

    def _test_accuracy(self, stats, events):
        pass

