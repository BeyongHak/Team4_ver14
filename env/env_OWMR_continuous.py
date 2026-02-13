import numpy as np
from collections import deque
import copy


class ManufacturerOWMREnv:
    def __init__(self, config, retailers):
        self.config = config
        self.retailers = retailers
        self.num_retailers = len(retailers)

        # Manufacturer Settings
        # [Mod] Normalization via SAA Policy Map (Global Max)
        # We replace heuristic max_inventory with the actual maximum target level from SAA.
        
        self.lead_time = config.get("lead_time", 2)
        
        # Load SAA Map & Meta for Normalization (Baseline Standard)
        saa_max_val = 0.0
        calculated_max_k = 50.0 # Default
        
        config_name = config.get("config_name", "") 
        
        import os
        import csv
        import ast
        
        # Try to locate SAA map relative to project root
        base_dir = os.getcwd() # Assumption: running from root
        baseline_dir = os.path.join(base_dir, "baseline")
        
        map_path = None
        meta_path = None
        
        if self.config.get("skip_normalization", False):
            # [User Request] Skip Normalization Logic
            # Used for SAA Policy Evaluation where Neural Network input is not needed.
            # Avoids looking for 'baseline' files that might not exist yet.
            print(f"[Env] Skipping Normalization/Baseline Loading (skip_normalization=True)")
            saa_max_val = 1.0 # Arbitrary non-zero value
            calculated_max_k = 50.0 # Default
            
            # Use simple defaults based on config if possible?
            is_homo = "Homo" in config_name or "D_Homo" in config_name
            if is_homo:
                calculated_max_k = [50.0] * self.num_retailers
            else:
                calculated_max_k = [50.0] * self.num_retailers
                
        else:
            # Standard DRL Normalization Logic
            if os.path.exists(baseline_dir):
                candidates = [f.replace(".csv", "") for f in os.listdir(baseline_dir) if f.endswith(".csv")]
                candidates.sort(key=len, reverse=True)
                
                for cand in candidates:
                    if cand in config_name:
                        map_path = os.path.join(baseline_dir, f"{cand}.csv")
                        meta_path = os.path.join(baseline_dir, f"{cand}_meta.txt")
                        break
            
            if map_path and os.path.exists(map_path):
                 # 1. Read Max Inventory from Meta (if exists)
                 if meta_path and os.path.exists(meta_path):
                     try:
                         with open(meta_path, 'r') as f:
                             # Read first line only for max_inventory to support extra metadata
                             line = f.readline()
                             if line:
                                saa_max_val = float(line.strip())
                         print(f"[Env] Loaded SAA Max from Meta: {saa_max_val} ({meta_path})")
                     except Exception as e:
                         print(f"[Env] Error reading Meta: {e}")
                 
                 # 2. Scan CSV for Max K
                 # [Mod] Differentiate Homo vs Hetero Max K Strategy
                 is_homo = "Homo" in config_name or "D_Homo" in config_name
                 
                 try:
                    scan_max_val = 0.0
                    
                    # Tracking Arrays
                    if is_homo:
                        max_k_tracker = 0 # Global Scalar
                    else:
                        max_k_tracker = [0] * self.num_retailers # Vector
                    if not os.path.exists(map_path): map_path = map_path.replace(".csv", "_policy.csv") # Try legacy name
                    
                    self.saa_states = [] # [New] Store SAA States for Reset Randomization
                    
                    with open(map_path, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            k_state = ast.literal_eval(row[0])
                            val = float(row[1])
                            
                            if val > scan_max_val: scan_max_val = val
                            
                            if isinstance(k_state, int): k_state = (k_state,)
                            else: k_state = tuple(int(x) for x in k_state)
                            
                            self.saa_states.append(k_state) # [New]
                            
                            if is_homo:
                                local_max = max(k_state)
                                if local_max > max_k_tracker: max_k_tracker = local_max
                            else:
                                # Hetero: Update per index
                                for i, k_v in enumerate(k_state):
                                    if i < len(max_k_tracker):
                                        if k_v > max_k_tracker[i]: max_k_tracker[i] = k_v
                    
                    # Finalize
                    if is_homo:
                        calculated_max_k = [float(max_k_tracker)] * self.num_retailers
                        print(f"[Env] Detected Homo Global Max K: {max_k_tracker}")
                    else:
                        calculated_max_k = [float(x) for x in max_k_tracker]
                        print(f"[Env] Detected Hetero Tensor Max K: {calculated_max_k}")
                    
                    # [Mod] Filter Base States (At least one zero)
                    self.base_states = [s for s in self.saa_states if 0 in s]
                    print(f"[Env] Pre-calculated Base States (Has 0): {len(self.base_states)} / {len(self.saa_states)}")

                    # Fallback if Meta was missing
                    if saa_max_val == 0.0 and scan_max_val > 0:
                        saa_max_val = scan_max_val
                        print(f"[Env] Fallback: Used Scanned Max Val {saa_max_val}")
                        
                 except Exception as e:
                     print(f"[Env] Error scanning map: {e}")
    
            else:
                print(f"[Env] Warning: Baseline not found for {config_name}. Using defaults.")

        
        # Set Global Max Inventory for Normalization
        self.max_inventory = saa_max_val 
        if self.max_inventory <= 1.0: self.max_inventory = 200.0 # Safety
        
        # [Mod] Continuous Action
        self.action_size = 1
        
        print(f"[Env] Final Max Inventory Norm: {self.max_inventory:.1f} (SAA: {saa_max_val})") 

        self.holding_cost = config.get("holding_cost", 1.0)
        self.backorder_cost = config.get("backorder_cost", 9.0)
        self.fixed_cost = config.get("fixed_cost", 0)
        self.max_steps = config.get("max_eps_length", 300)
        self.reward_scale = config.get("reward_scale", 100.0)
        
        # State Dimensions
        self.state_size = 1 + 1 + self.lead_time + self.num_retailers

        # [Mod] Retailer max K from Baseline Scan (List)
        self.retailer_max_k = calculated_max_k

        self.m_IL = 0.0
        self.m_IP = 0.0

        if self.lead_time > 0:
            self.m_pipeline = deque([0.0] * self.lead_time, maxlen=self.lead_time)
        else:
            self.m_pipeline = None 

        self.r_IP = np.zeros(self.num_retailers)
        self.r_order = np.zeros(self.num_retailers)
        self.r_k_states = np.zeros(self.num_retailers, dtype=int)

        self.current_step = 0
        
        # [Restored] Fixed Demand Scenario Support for Evaluator
        self.fixed_demands = None 
        self.fixed_demand_ptr = 0

    def set_fixed_demands(self, demands):
        """
        Set a fixed sequence of demands for evaluation.
        demands: list or array of shape (N_steps, Num_Retailers)
        """
        self.fixed_demands = demands
        self.fixed_demand_ptr = 0

    def set_max_steps(self, max_steps):
        """
        Dynamically update max steps for evaluation scenarios.
        """
        self.max_steps = int(max_steps)

    def reset(self):
        self.current_step = 0
        self.cumulative_reward = 0.0
        
        # [Common] Manufacturer Always Zero Init (User Request)
        self.m_IL = 0.0 
        self.m_IP = 0.0
        if self.lead_time > 0:
            self.m_pipeline = deque([0.0] * self.lead_time, maxlen=self.lead_time)
        else:
            self.m_pipeline = None 

        # Determine Mode
        is_eval_mode = (self.fixed_demands is not None) or self.config.get("eval_mode", False)

        if is_eval_mode:
            # [Evaluation/Numerical] Deterministic Zero Init
            self.r_IP = np.array([r['delta'] for r in self.retailers], dtype=float)
            self.r_k_states = np.zeros(self.num_retailers, dtype=int)
            
            # Reset pointer if using fixed demands
            if self.fixed_demands is not None:
                 self.fixed_demand_ptr = 0
                
        else:
            # [Training] Randomized Initialization
            # Experience varied retailer states to learn recovery/maintenance
            if hasattr(self, 'base_states') and len(self.base_states) > 0:
                # [User Request] Sample random BASE state (at least one 0)
                idx = np.random.randint(0, len(self.base_states))
                
                # Handles Tuple vs List
                selected_state = self.base_states[idx]
                self.r_k_states = np.array(selected_state, dtype=int)
                
                # Derive r_IP from r_k (Heuristic: IP = Delta - Consumed)
                self.r_IP = np.zeros(self.num_retailers)
                for i, r in enumerate(self.retailers):
                    if self.r_k_states[i] > 0:
                        # [Stochastic] Sample 'k' demand realizations to estimate consumption
                        # This captures the variance of the demand process better than (k * mean)
                        k_val = int(self.r_k_states[i])
                        sampled_demands = r['customer'].rvs(size=k_val)
                        consumed = np.sum(sampled_demands)
                    else:
                        consumed = 0.0
                        
                    val = r['delta'] - consumed
                    # Ensure minimal IP to avoid immediate stockout crash if k is large
                    # Clamping to 0 effectively means we are at the brink of stockout
                    self.r_IP[i] = max(val, 0.0) 
            else:
                # [Debugging] Strict Mode: SAA states MUST be loaded for training
                raise ValueError(f"[Env] SAA States NOT loaded for randomized initialization! \n"
                                 f"Please check if SAA Policy Map exists and is correctly loaded in __init__.")

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
            # Agent outputs the physical Target Level (S_RL = S_SAA * ratio).
            # No scaling/clipping to [-1, 1] required.
            # ----------------------------------------------------------------
            target_level = float(action)
            
            # Calculate Physical Order Quantity
            # Order = max(0, Target - IP)
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
                    order_qty = retailer['delta'] - self.r_IP[i]
                    self.r_order[i] = order_qty 
                    self.r_IP[i] = retailer['delta'] 
                    self.r_k_states[i] = 0 
                else:
                    self.r_order[i] = 0
                    self.r_k_states[i] += 1

                total_retailer_orders += self.r_order[i]
                
                # [Mod] Demand Generation (Random vs Fixed)
                if self.fixed_demands is not None:
                    # Use fixed demand if available and within range
                    if self.fixed_demand_ptr < len(self.fixed_demands):
                        c_demand = self.fixed_demands[self.fixed_demand_ptr][i]
                    else:
                        # Fallback to random to prevent crash
                        c_demand = retailer['customer'].rvs()
                else:
                    c_demand = retailer['customer'].rvs()

                self.r_IP[i] -= c_demand

            # Increment ptr after processing all retailers for this step
            if self.fixed_demands is not None:
                self.fixed_demand_ptr += 1

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
                "target_level": target_level,  # 실제 타겟 레벨 (-50 ~ 500)
                "retailer_orders": total_retailer_orders,
                "retailer_states": self.r_k_states.copy()
            }

            return self._get_state(), reward, done, info

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