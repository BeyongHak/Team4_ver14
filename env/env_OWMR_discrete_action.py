import numpy as np
from collections import deque
import copy


class ManufacturerOWMRDiscreteActionEnv:
    def __init__(self, config, retailers):
        self.config = config
        self.retailers = retailers
        self.num_retailers = len(retailers)

        # Manufacturer Settings
        # [Mod] Normalization via SAA Policy Map (Global Max)
        self.lead_time = config.get("lead_time", 2)
        
        # Load SAA Map to find Global Max
        saa_max_val = 0.0
        config_name = config.get("config_name", "") 
        
        import os
        import csv
        
        # Try to locate SAA map relative to project root
        base_dir = os.getcwd() # Assumption: running from root
        policy_saa_dir = os.path.join(base_dir, "policy_SAA")
        
        found_saa = False
        if os.path.exists(policy_saa_dir):
            target_config = config.get("config_name", "")
            if target_config:
                # Search Logic
                for d in os.listdir(policy_saa_dir):
                    if d in target_config:
                        cand = os.path.join(policy_saa_dir, d, "SAA_policy_map.csv")
                        if os.path.exists(cand):
                            # Load and find Max
                            try:
                                max_v = 0.0
                                with open(cand, 'r') as f:
                                    reader = csv.reader(f)
                                    next(reader)
                                    for row in reader:
                                        val = float(row[2])
                                        if val > max_v: max_v = val
                                saa_max_val = max_v
                                found_saa = True
                                print(f"[Env-Discrete] Loaded SAA Max for Normalization: {saa_max_val} from {d}")
                            except:
                                pass
                        if found_saa: break
        
        if not found_saa:
            # [Strict] Raise Error if SAA Map is missing
            raise FileNotFoundError(f"[Env-Discrete] SAA Policy Map not found for config: '{config_name}'. "
                                    f"Normalization requires this map. Please run SAA_LeadtimeDemand.py first.")
        
        # Set Global Max Inventory for Normalization
        # [Mod] Use SAA Max directly (User Request).
        # [Fix] Apply action_buffer_factor to Normalization Range
        buffer_factor = config.get("action_buffer_factor", 1.0)
        self.max_inventory = saa_max_val * buffer_factor
        
        # [Mod] Continuous Action
        self.action_size = 1
        
        # Removed min_action legacy
        
        print(f"[Env-Discrete] Final Max Inventory for Norm: {self.max_inventory:.1f} (SAA: {saa_max_val}, Buffer: {buffer_factor})") 
        
        self.holding_cost = config.get("holding_cost", 1.0)
        self.backorder_cost = config.get("backorder_cost", 9.0)
        self.fixed_cost = config.get("fixed_cost", 0)
        self.max_steps = config.get("max_eps_length", 300)
        self.reward_scale = config.get("reward_scale", 100.0)

        # State Dimensions
        # M_IL, M_B, M_pipeline, R_k_state, R_k_order
        self.state_size = 1 + 1 + self.lead_time + self.num_retailers
        
        # Retailer max K for normalization
        self.retailer_max_k = [getattr(r, 'K', 50.0) for r in self.retailers]

        self.m_IL = 0.0
        self.m_IP = 0.0

        if self.lead_time > 0:
            self.m_pipeline = deque([0.0] * self.lead_time, maxlen=self.lead_time)
        else:
            self.m_pipeline = None 

        self.r_IP = np.zeros(self.num_retailers)
        self.r_order = np.zeros(self.num_retailers)
        self.r_k_states = np.zeros(self.num_retailers, dtype=int)
        
        # [Restored] Fixed Demand Scenario Support for Evaluator
        self.fixed_demands = None
        self.fixed_demand_ptr = 0

        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.cumulative_reward = 0.0
        
        self.m_IL = 0.0 
        self.m_IP = 0.0

        if self.lead_time > 0:
            self.m_pipeline = deque([0.0] * self.lead_time, maxlen=self.lead_time)
        else:
            self.m_pipeline = None 

        self.r_IP = np.array([r.delta for r in self.retailers], dtype=float)
        self.r_k_states = np.zeros(self.num_retailers, dtype=int)
        
        # Reset pointer if using fixed demands
        if self.fixed_demands is not None:
             self.fixed_demand_ptr = 0
        
        return self._get_state()

    def _get_state(self):
        """
        State Normalization
        """
        raw_k = self.r_k_states.astype(np.float32)
        norm_k = []
        
        on_hand = max(0, self.m_IL)
        backorder = max(0, -self.m_IL)
        inventory_status = [on_hand, backorder]        
        
        if self.lead_time > 0 :
            inventory_status.extend(self.m_pipeline)

        inventory_status = np.array(inventory_status, dtype=np.float32)

        # [Mod] Normalize Inventory Status by Max Inventory
        inventory_status = inventory_status / self.max_inventory

        for i, k_val in enumerate(raw_k):
            limit = self.retailer_max_k[i]
            norm_k.append((k_val / limit)) 

        total_state = np.concatenate((inventory_status, norm_k), dtype=np.float32)

        return total_state

    def step(self, action):
            # ----------------------------------------------------------------
            # 1. Manufacturer Action (Ordering)
            # [Mod] Direct Target Level Input (Beta Distribution Logic)
            # ----------------------------------------------------------------
            target_level = float(action)
            
            # [Mod] Round to nearest integer for Discrete Action (Environment limit)
            target_level = int(np.round(target_level))
            
            m_order = max(0.0, target_level - self.m_IP)
            self.m_IP += m_order

            # ----------------------------------------------------------------
            # 2. Pipeline Processing (Arrival of Goods)
            # ----------------------------------------------------------------
            arrived_goods = 0.0
            if self.lead_time > 0:
                arriving_item = self.m_pipeline[0]
                self.m_pipeline.append(m_order)
                arrived_goods = arriving_item
            else:
                arrived_goods = m_order

            self.m_IL += arrived_goods

            # ----------------------------------------------------------------
            # 3. Retailer Logic
            # ----------------------------------------------------------------
            total_retailer_orders = 0.0 

            for i, retailer in enumerate(self.retailers):
                if self.r_IP[i] <= 0:
                    order_qty = retailer.delta - self.r_IP[i]
                    self.r_order[i] = order_qty 
                    self.r_IP[i] = retailer.delta 
                    self.r_k_states[i] = 0 
                else:
                    self.r_order[i] = 0
                    self.r_k_states[i] += 1

                total_retailer_orders += self.r_order[i]
                
                # Consume Demand (Fixed or Random)
                if self.fixed_demands is not None:
                    c_demand = self.fixed_demands[self.current_step][i]
                else:
                    c_demand = retailer.customer.rvs()
                    
                self.r_IP[i] -= c_demand

            # ----------------------------------------------------------------
            # 4. Manufacturer Demand Satisfaction & Cost Calc
            # ----------------------------------------------------------------
            self.m_IL -= total_retailer_orders
            self.m_IP -= total_retailer_orders 

            # Cost Calculation
            cost = 0.0
            if self.m_IL >= 0:
                cost += self.holding_cost * self.m_IL
            else:
                cost += self.backorder_cost * (-self.m_IL)
            
            cost += self.fixed_cost
            reward = -cost / self.reward_scale

            # ----------------------------------------------------------------
            # 5. Finish & Return
            # ----------------------------------------------------------------
            self.cumulative_reward += reward
            self.current_step += 1
            done = 1 if self.current_step >= self.max_steps else 0

            info = {
                "m_IL": self.m_IL,
                "m_IP": self.m_IP,
                "order_qty": m_order,          
                "target_level": target_level,  # 실제 타겟 레벨 (Rounded)
                "retailer_orders": total_retailer_orders,
                "retailer_states": self.r_k_states.copy()
            }

            return self._get_state(), reward, done, info
    
    def set_fixed_demands(self, demands):
        """
        Set fixed demand patterns for evaluation (Scenario).
        demands: numpy array or list of shape (episode_len, num_retailers)
        """
        self.fixed_demands = demands
        self.fixed_demand_ptr = 0

    def getStateSize(self):
        return self.state_size

    def getActionSize(self):
        return self.action_size

    def getValidActions(self, state):
        mask = np.ones(self.action_size, dtype=np.float32)
        return mask

    def close(self):
        """Clean up resources if any."""
        pass

    def run(self, agent, num_games):
        total_cumulative_reward = 0.0
        for game_idx in range(num_games):
            state = self.reset()
            done = 0
            while not done:
                action, step_data = agent.play(state)
                next_state, reward, done, info = self.step(action)
                if step_data is not None:
                    agent.rollout.add_data(
                        state        = step_data["state"],
                        action       = action,
                        value        = step_data["value"],
                        reward       = reward, 
                        done         = done,
                        valid_action = step_data["valid_action"],
                        prob         = step_data["prob"],
                        memory       = step_data["memory"],
                        policy       = step_data["policy"]
                    )
                
                if done:
                    agent.rollout.game_count += 1
                    agent.rollout.step_count = 0
                else:
                    agent.rollout.step_count += 1
                    max_step = agent.rollout.max_eps_length - 1
                    if agent.rollout.step_count > max_step:
                        agent.rollout.step_count = max_step

                state = next_state
            total_cumulative_reward += self.cumulative_reward
       
        avg_reward = total_cumulative_reward / num_games
        return avg_reward
