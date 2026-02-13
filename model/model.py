import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
import ast

class PPOMLPModel(nn.Module):
    def __init__(self, config, state_size, action_size):
        super(PPOMLPModel, self).__init__()
        
        self.lead_time = config["lead_time"]
        self.num_retailers = len(config["retailers"])
        self.config = config  # [Fix] Store config for later access (get_saa_base_action)
        
        # Load SAA Policy Map from 'baseline' directory
        self.saa_map = {}
        config_name = config.get("config_name", "") # Needs to be injected in main.py
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root from model/model.py
        baseline_dir = os.path.join(base_dir, "baseline")
        
        map_path = None
        meta_path = None
        
        # Search Strategy: Find matching key in baseline
        # Candidates: Check if any file in baseline matches the config_name
        # config_name e.g. "Homo_5R_8_0_1" -> baseline "Homo_5R_8"
        
        if os.path.exists(baseline_dir):
            candidates = [f.replace(".csv", "") for f in os.listdir(baseline_dir) if f.endswith(".csv")]
            # Sort by length descending to match longest prefix
            candidates.sort(key=len, reverse=True)
            
            for cand in candidates:
                if cand in config_name:
                    map_path = os.path.join(baseline_dir, f"{cand}.csv")
                    meta_path = os.path.join(baseline_dir, f"{cand}_meta.txt")
                    break
        
        # Fallback Logic (Legacy) if not found in baseline? 
        # User requested to use baseline. We warn if not found.
        
        # 1. Determine Identity (Homo vs Hetero)
        self.is_homo = "Homo" in config_name or "D_Homo" in config_name 
        # Fallback if config_name not set
        if not config_name and "retailers" in config:
             # Basic heuristic: check if all retailers identical?
             # For now assume Homo if not explicitly Hetero? No, safer to default Hetero if unknown.
             pass
        
        # 2. Scanning Baseline
        print(f"[Model] Config: {config_name} | Type: {'Homogeneous' if self.is_homo else 'Heterogeneous'}")
        
        if map_path and os.path.exists(map_path):
            print(f"[Model] Loading SAA Policy Map from {map_path}")
            
            # Tracking Max K
            # Homo: Single Global Max
            # Hetero: Max per Retailer
            if self.is_homo:
                global_max_k = 0
            else:
                current_max_ks = [0] * self.num_retailers

            with open(map_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) # Skip Header
                for row in reader:
                    try:
                        k_state = ast.literal_eval(row[0])
                        if isinstance(k_state, int): k_state = (k_state,)
                        else: k_state = tuple(int(x) for x in k_state)
                        
                        val = float(row[1]) 
                        
                        # Max K Tracking
                        if self.is_homo:
                            local_max = max(k_state)
                            if local_max > global_max_k: global_max_k = local_max
                            
                            # Store Map (Key is SORTED state for Homo)
                            # Assuming baseline might be unsorted? Usually baseline is canonical.
                            # But to be safe, we sort key.
                            key_store = tuple(sorted(k_state))
                            self.saa_map[key_store] = val
                            
                        else:
                            # Hetero: Track per index
                            for i, k_val in enumerate(k_state):
                                if i < self.num_retailers:
                                    if k_val > current_max_ks[i]: current_max_ks[i] = k_val
                            
                            # Store Map (Key is RAW state)
                            self.saa_map[k_state] = val
                            
                    except Exception:
                        pass
            
            # Finalize Max K Tensor
            if self.is_homo:
                 print(f"[Model] Determined Global Max K (Homo): {global_max_k}")
                 self.retailer_max_k_tensor = torch.tensor([global_max_k] * self.num_retailers, dtype=torch.float32)
            else:
                 print(f"[Model] Determined Per-Retailer Max K (Hetero): {current_max_ks}")
                 self.retailer_max_k_tensor = torch.tensor(current_max_ks, dtype=torch.float32)

        else:
            print(f"[Warning] SAA Policy Map NOT FOUND for {config_name}. Defaults used.")
            if self.is_homo:
                self.retailer_max_k_tensor = torch.tensor([10.0] * self.num_retailers, dtype=torch.float32)
            else:
                self.retailer_max_k_tensor = torch.tensor([10.0] * self.num_retailers, dtype=torch.float32)

        # [New] Calculate Default Fallback Value
        if self.saa_map:
            self.default_saa_val = max(self.saa_map.values())
        else:
            self.default_saa_val = 0.0

        # Input Dimensions
        self.inv_dim = 2 + self.lead_time
        self.ret_dim = self.num_retailers
        
        # [New] Network Architecture Configuration
        net_config = config.get("network_arch", {})
        
        # Default Settings
        default_hidden = 128
        default_embed  = 64
        
        self.embedding_dim = net_config.get("embedding_dim", default_embed)
        self.embedding_dim_A = net_config.get("embedding_dim_A", self.embedding_dim)
        self.embedding_dim_C = net_config.get("embedding_dim_C", self.embedding_dim)
        
        self.mlp_inv_arch = net_config.get("mlp_inv_layers", [default_hidden])
        self.mlp_ret_arch = net_config.get("mlp_ret_layers", [default_hidden])
        self.policy_arch  = net_config.get("policy_layers", [default_hidden, default_hidden])
        self.value_arch   = net_config.get("value_layers", [default_hidden, default_hidden, default_hidden])
        
        act_name = config.get("activation", "tanh").lower()
        if act_name == "relu":
            self.activation_cls = nn.ReLU
        else:
            self.activation_cls = nn.Tanh 

        # Build Networks
        self.mlp_C = self._build_mlp(
            input_dim=self.inv_dim + self.ret_dim, 
            hidden_layers=self.mlp_inv_arch, 
            output_dim=self.embedding_dim_C, 
            hidden_activation_cls=self.activation_cls,
            final_activation=self.activation_cls() 
        )

        self.mlp_A = self._build_mlp(
            input_dim=self.ret_dim, 
            hidden_layers=self.mlp_ret_arch, 
            output_dim=self.embedding_dim_A, 
            hidden_activation_cls=self.activation_cls,
            final_activation=self.activation_cls() 
        )

        self.policy = self._build_mlp(
            input_dim=self.embedding_dim_A, 
            hidden_layers=self.policy_arch, 
            output_dim=action_size * 2, 
            hidden_activation_cls=self.activation_cls,
            final_activation=None, 
            init_scale=0.01
        )
        
        last_actor_layer = self.policy[-2] if isinstance(self.policy[-1], (nn.ReLU, nn.Tanh)) else self.policy[-1]
        
        if isinstance(last_actor_layer, nn.Linear):
            # [Mod] Init Mean to ~10 (Softplus(10) ~ 10)
            # Inverse Softplus(10): log(exp(10)-1) ~ 10.0
            last_actor_layer.bias.data[:action_size].fill_(0.0) 
            
            # [Mod] Init Std to ~1.0 (exp(log_std) = 1 -> log_std = 0.0)
            last_actor_layer.bias.data[action_size:].fill_(0.0)
            print(f"[Model] Initialized Gaussian Policy: Mean~10.0, Std~1.0")

        self.value = self._build_mlp(
            input_dim=self.embedding_dim_C, 
            hidden_layers=self.value_arch, 
            output_dim=1, 
            hidden_activation_cls=self.activation_cls,
            final_activation=None,
            init_scale=1.0
        )

    def _build_mlp(self, input_dim, hidden_layers, output_dim, hidden_activation_cls=nn.Tanh, final_activation=None, init_scale=1.0):
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(self._layer_init(nn.Linear(curr_dim, h_dim), std=np.sqrt(2)))
            layers.append(hidden_activation_cls())
            curr_dim = h_dim
        
        layers.append(self._layer_init(nn.Linear(curr_dim, output_dim), std=init_scale))
        if final_activation:
            layers.append(final_activation)
            
        return nn.Sequential(*layers)

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, state):
        """
        state: (Batch, State_Dim)
        """
        # Split state
        inv_input = state[:, :self.inv_dim]
        ret_input = state[:, self.inv_dim:]
        
        # [Sorting Strategy]
        # If Homogeneous, sort retailer inputs to ensure Permutation Invariance.
        # This helps the model learn a canonical representation.
        if self.is_homo:
            ret_sorted, _ = torch.sort(ret_input, dim=1, descending=False)
            
            # Reconstruct State with Sorted Retailers
            # Used for Critic and Actor
            proc_ret_input = ret_sorted
            proc_state = torch.cat([inv_input, ret_sorted], dim=1)
        else:
            proc_ret_input = ret_input
            proc_state = state
            
        # Embeddings
        emb_c = self.mlp_C(proc_state)
        emb_a = self.mlp_A(proc_ret_input) 
        
        # Actor
        policy_out = self.policy(emb_a) 
        mean, log_std = torch.chunk(policy_out, 2, dim=-1)
        
        # [Mod] Enforce positive mean via Softplus (User Request)
        mean = F.softplus(mean)
        
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        # Critic
        value = self.value(emb_c)

        return (mean, log_std), value

    def get_saa_base_action(self, state):
        """
        Lookup SAA Base Stock Level for the current state.
        state: (Batch, State_Dim)
        Returns: (Batch, Action_Dim)
        """
        # Extract Retailer State
        ret_input = state[:, self.inv_dim:] # (Batch, Num_Retailers)
        
        # Denormalize to Integers
        # (Batch, R) * (R,)
        # Note: retailer_max_k_tensor is already correctly set (Global Max for Homo, Per-Ret for Hetero)
        max_k_tensor = self.retailer_max_k_tensor.to(state.device)
        
        # Norm [0,1] -> [0, Max]
        raw_k = (ret_input * max_k_tensor).round().long()
        
        # [Safety] Clamp index to Max K (Nearest Neighbor Lookup)
        # If state > Max K (Extrapolation), use action for Max K (which is usually 0).
        # This prevents "KeyError" fallback to arbitrary default.
        # [Safety] Clamp index to Max K (Nearest Neighbor Lookup)
        # Replacing torch.clamp(max=Tensor) with torch.min(Tensor) for backward compatibility/broadcasting.
        # 1. Clamp Min (Scalar 0)
        raw_k = torch.clamp(raw_k, min=0)
        # 2. Clamp Max (Tensor Broadcast)
        raw_k = torch.min(raw_k, max_k_tensor.long())
        
        batch_size = state.shape[0]
        saa_actions = torch.zeros(batch_size, 1, device=state.device) 
        
        raw_k_cpu = raw_k.cpu().numpy()
        
        for i in range(batch_size):
            # Prepare Key
            current_k = raw_k_cpu[i]
            
            if self.is_homo:
                # [Sorting Strategy] Homo: Sort key before lookup
                # saa_map keys were sorted during loading
                key = tuple(sorted(current_k))
            else:
                # Hetero: Use raw key
                key = tuple(current_k)
            
            # Lookup
            if key in self.saa_map:
                val = self.saa_map[key]
            else:
                # Fallback (Should be rare if map is complete)
                val = self.default_saa_val 
            
            # Apply Action Buffer Factor
            buffer_factor = self.config.get("action_buffer_factor", 1.0)
            val = val * buffer_factor
            
            saa_actions[i] = val
            
        return saa_actions

    
