from typing import List, Tuple
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.torch_entities.action_model import ActionModel

import itertools

def actions_combine(
    actions: List[AgentAction]
) -> AgentAction:
    continuous_tensor, discrete_list = None, None

    continuous_tensors = list(map(lambda a: a.continuous_tensor, actions))
    if any(map(lambda t: t is not None, continuous_tensors)):
        continuous_tensor = torch.cat(continuous_tensors, dim=1)

    discrete_lists = list(map(lambda a: a.discrete_list, actions))
    if any(map(lambda t: t is not None, discrete_lists)):
        discrete_list = list(itertools.chain.from_iterable(discrete_lists))

    return AgentAction(continuous_tensor, discrete_list)

def action_probs_combine(
    action_probs: List[ActionLogProbs] 
) -> ActionLogProbs:
    continuous_tensor, discrete_list, all_discrete_list = None, None, None

    continuous_tensors = list(map(lambda a: a.continuous_tensor, action_probs))
    if any(map(lambda t: t is not None, continuous_tensors)):
        continuous_tensor = torch.cat(continuous_tensors, dim=1)

    discrete_lists = list(map(lambda a: a.discrete_list, action_probs))
    if any(map(lambda t: t is not None, discrete_lists)):
        discrete_list = list(itertools.chain.from_iterable(discrete_lists))
    
    all_discrete_lists = list(map(lambda a: a.all_discrete_list, action_probs))
    if any(map(lambda t: t is not None, all_discrete_lists)):
        all_discrete_list = list(itertools.chain.from_iterable(all_discrete_lists))

    return ActionLogProbs(continuous_tensor, discrete_list, all_discrete_list)

def get_actions_subset(
    actions: List[AgentAction], model: ActionModel, continuous_actions_offset: int, discrete_actions_offset: int
)-> Tuple[int, int, AgentAction]:
    continuous_tensor, discrete_list = None, None
    
    continuous_spec = model.action_spec.continuous_size
    if continuous_spec != 0:
        continuous_tensor = actions.continuous_tensor[:, continuous_actions_offset:continuous_actions_offset+continuous_spec]
        continuous_actions_offset = continuous_actions_offset + continuous_spec

    discrete_spec = model.action_spec.discrete_size
    if discrete_spec != 0:
        discrete_list = actions.discrete_list[discrete_actions_offset:discrete_actions_offset+discrete_spec]
        discrete_actions_offset = discrete_actions_offset + discrete_spec

    action = AgentAction(continuous_tensor, discrete_list)
    return (action, continuous_actions_offset, discrete_actions_offset)

class AggregatedActionModel(nn.Module):
    def __init__(
        self,
        models: List[ActionModel],
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
        deterministic: bool = False
    ):
        """
        A torch module that represents the combination of action spaces 
        represented by action models used by each of the actuators
        """
        super().__init__()
        self.encoding_size = sum(list(map(lambda m: m.encoding_size, models)))
        self.action_spec = action_spec
        self.models = nn.ModuleList(models)

        self.clip_action = not tanh_squash
        self._deterministic = deterministic

    def evaluate(
        self, inputs: torch.Tensor, masks: torch.Tensor, actions: AgentAction
    ) -> Tuple[ActionLogProbs, torch.Tensor]:
        """
        Given actions and encoding from the network body, gets the distributions and
        computes the log probabilites and entropies.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :params actions: The AgentAction
        :return: An ActionLogProbs tuple and a torch tensor of the distribution entropies.
        """        
        results = []
        inputs_offset = 0
        continuous_actions_offset = 0
        discrete_actions_offset = 0

        for model in self.models:
            # Slice inputs, masks and actions according to action model signature
            inputs_slice = inputs[:, inputs_offset:inputs_offset + model.encoding_size]
            masks_slice = masks[:, inputs_offset:inputs_offset + model.encoding_size] if masks is not None else masks
            inputs_offset = inputs_offset + model.encoding_size

            actions_slice, continuous_actions_offset, discrete_actions_offset = get_actions_subset(
                actions, model, continuous_actions_offset, discrete_actions_offset
            )

            results.append(model.evaluate(inputs_slice, masks_slice, actions_slice))
        
        action_probs = list(map(lambda t: t[0], results))
        log_probs = action_probs_combine(action_probs)

        entropy_sums = list(map(lambda t: t[1], results))
        entropy_sum = torch.stack(entropy_sums, dim=1).sum(dim=1)
        
        return log_probs, entropy_sum

    def get_action_out(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Gets the tensors corresponding to the output of the policy network to be used for
        inference. Called by the Actor's forward call.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :return: A tuple of torch tensors corresponding to the inference output
        """
        
        results = []
        inputs_offset = 0

        for model in self.models:
            # Slice inputs and masks according to action model signature
            inputs_slice = inputs[:, inputs_offset:inputs_offset + model.encoding_size]
            masks_slice = masks[:, inputs_offset:inputs_offset + model.encoding_size] if masks is not None else masks
            inputs_offset = inputs_offset + model.encoding_size

            results.append(model.get_action_out(inputs_slice, masks_slice))

        continuous_out, discrete_out, action_out_deprecated, deterministic_continuous_out, deterministic_discrete_out = None, None, None, None, None

        # Combine the results
        continuous_outs = list(map(lambda r: r[0], results))
        if any(map(lambda t: t is not None, continuous_outs)):
            continuous_out = torch.cat(continuous_outs, dim=1)
        
        discrete_outs = list(map(lambda r: r[1], results))
        if any(map(lambda t: t is not None, discrete_outs)):
            discrete_out = torch.cat(discrete_outs, dim=1)

        actions_out_deprecated = list(map(lambda r: r[2], results))
        if any(map(lambda t: t is not None, actions_out_deprecated)):
            action_out_deprecated = torch.cat(actions_out_deprecated, dim=1)

        deterministic_continuous_outs = list(map(lambda r: r[3], results))
        if any(map(lambda t: t is not None, deterministic_continuous_outs)):
            deterministic_continuous_out = torch.cat(deterministic_continuous_outs, dim=1)

        deterministic_discrete_outs = list(map(lambda r: r[4], results))
        if any(map(lambda t: t is not None, deterministic_discrete_outs)):
            deterministic_discrete_out = torch.cat(list(map(lambda r: r[4], results)), dim=1)

        return (
            continuous_out,
            discrete_out,
            action_out_deprecated,
            deterministic_continuous_out,
            deterministic_discrete_out,
        )

    def forward(
        self, inputs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor]:
        """
        The forward method of this module. Outputs the action, log probs,
        and entropies given the encoding from the network body.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :return: Given the input, an AgentAction of the actions generated by the policy and the corresponding
        ActionLogProbs and entropies.
        """
        
        results = []
        inputs_offset = 0

        for model in self.models:
            # Slice inputs, masks and actions according to action model signature
            inputs_slice = inputs[:, inputs_offset:inputs_offset + model.encoding_size]
            masks_slice = masks[:, inputs_offset:inputs_offset + model.encoding_size] if masks is not None else masks
            inputs_offset = inputs_offset + model.encoding_size

            results.append(model.forward(inputs_slice, masks_slice))
        
        actions_list = list(map(lambda t: t[0], results))
        actions = actions_combine(actions_list)

        action_probs = list(map(lambda t: t[1], results))
        log_probs = action_probs_combine(action_probs)

        entropy_sums = list(map(lambda t: t[2], results))
        entropy_sum = torch.stack(entropy_sums, dim=1).sum(dim=1)
        
        return (actions, log_probs, entropy_sum)