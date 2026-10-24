"""Unit tests for Dynamic Pricing Agents."""

import numpy as np
from src.agents.baselines import FixedCatalogPricingAgent, RuleBasedMarkdownAgent
from src.agents.q_learning import QLearningPricingAgent
from src.agents.bandit_agents import LinUCBPricingAgent


def test_baseline_agents():
    obs = np.array([0.8, 0.9, 0.5, 1.0, 0.0, 1.2], dtype=np.float32)

    fixed_agent = FixedCatalogPricingAgent(action_index=3, num_actions=7)
    assert fixed_agent.select_action(obs) == 3

    markdown_agent = RuleBasedMarkdownAgent(num_actions=7)
    action = markdown_agent.select_action(obs)
    assert 0 <= action < 7


def test_q_learning_agent_step_and_update():
    agent = QLearningPricingAgent(num_actions=7, seed=42)
    obs = np.array([0.7, 0.8, 0.4, 1.0, 0.0, 1.0], dtype=np.float32)
    next_obs = np.array([0.6, 0.78, 0.4, 1.0, 0.0, 1.5], dtype=np.float32)

    action = agent.select_action(obs, evaluate=False)
    assert 0 <= action < 7

    initial_q_len = len(agent.q_table)
    agent.update(obs, action, reward=50000.0, next_state=next_obs, done=False)
    assert len(agent.q_table) >= initial_q_len


def test_linucb_agent():
    agent = LinUCBPricingAgent(context_dim=6, num_actions=7)
    obs = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 2.0], dtype=np.float32)

    action = agent.select_action(obs, evaluate=False)
    assert 0 <= action < 7

    agent.update(obs, action, reward=75000.0, next_state=obs, done=False)
