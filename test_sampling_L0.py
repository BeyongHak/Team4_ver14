import numpy as np
import os
import shutil
from env.manufacturer import ManufacturerMultiRetailer
from env.retailer import Retailer
from env.truncNormal import HybridTruncNorm
import matplotlib.pyplot as plt

def test_sampling_vs_theory_L0():
    print("=== Test Sampling vs Theory (L=0) ===")
    
    # 1. Setup Environment
    l = 0
    mean_val = 50.0
    std_val = 10.0
    delta = 150
    
    # Create Directories
    os.makedirs('leadtime demand', exist_ok=True)
    os.makedirs('policy', exist_ok=True)
    os.makedirs('retailer', exist_ok=True)
    os.makedirs('manufacturer', exist_ok=True)
    os.makedirs('retailer demand sample', exist_ok=True)

    # Create Customer with Truncated Normal
    customer = HybridTruncNorm(mean=mean_val, std=std_val, lower=0)
    
    # Create Retailer
    retailer = Retailer(delta=delta, customer=customer)
    
    # Create Manufacturer (Homo, N=1)
    # This will trigger _leadtime_demand_joint
    
    # Ensure clean state (delete old pickle if exists)
    pkl_name = f'leadtime demand/leadtime_demand_multi_pomrd l(0) K({retailer.K}) n(1) I(Homo)_sampling.pkl'
    if os.path.exists(pkl_name):
        os.remove(pkl_name)
    
    print(f"Creating Manufacturer with L={l}...")
    # Mocking K_list for single retailer since we pass list of retailers
    manufacturer = ManufacturerMultiRetailer(l=l, beta=0.9, retailers=[retailer], is_identical='Homo')
    
    # 2. Get Generated Leadtime Demand
    Dl_list = manufacturer.Dl_joint_list
    
    # 3. Compare with Theory
    # For L=0, state k=0 (Just Ordered), the demand seen by Manufacturer 
    # should be the Customer Demand Distribution (since Order = Demand to restore position)
    # Wait, if k=0 (Just ordered), IP becomes S. 
    # Demand seen by Manuf is the Order quantity to bring IP back to S.
    # Order_qty = Demand of previous period? 
    # Leadtime Demand for Manuf is "Sum of Orders from t to t+L".
    # If L=0, it's just "Order at time t".
    
    # Let's check state k=0.
    # In sampling logic:
    # Init IP = S (full).
    # t=0 (L+1 loop, size 1):
    #   Order if IP <= s (0). S (150) <= 0 is False. No Order.
    #   Wait, if we start with S, we don't order at t=0 unless we simulate one period demand first?
    #   The loop is `for t in range(l+1)`. If l=0, range(1). t=0.
    #   sim_ip starts at S.
    #   Order = max(sim_ip <= s(=0) ? S-sim_ip : 0, 0).
    #   Since sim_ip=S=150 > 0, Order=0.
    #   Then sim_ip -= demand.
    #   So Order is 0?
    
    #   Something is wrong. Leadtime Demand of 0 means "Demand during 0 leadtime"? 
    #   Usually Leadtime Demand is demand during L+1 periods (review period + leadtime)?
    #   In `main.py`, manufacturer uses `l=0` config.
    #   Standard theory: Order up to S covering D during L+1.
    #   If our sampling produces 0 orders for L=0, then Base Stock will be 0.
    
    #   Let's check the logic:
    #   BaseStock Policy: Order UP TO S. 
    #   The decision is made at time t. Order arrives at t+L.
    #   We want to protect against demand from t to t+L.
    #   So we need stats of "Demand from t to t+L".
    #   But "Demand" here is CONSUMER DEMAND? OR RETAILER ORDER?
    #   For Manufacturer, the "Demand" is Retailer Orders.
    #   So we need distribution of "Retailer Orders from t to t+L".
    
    #   If k=0 (Retailer just ordered to S), then at time t, Retailer IP=S.
    #   Retailer won't order until IP drops below s=0.
    #   With mean=50, delta=150, s=0:
    #   It takes ~3 periods to drop below 0.
    #   So for t=0, 1, 2... orders are likely 0.
    
    #   So for L=0 (1 period horizon), if k=0, Expected Order is 0. 
    #   This implies Manufacturer shouldn't hold stock if Retailer is full. Correct.
    
    #   Now check state k where Retailer is LOW.
    #   State k implies "k periods since last order".
    #   If k is large (e.g. 2 or 3), IP is low.
    #   Likelihood of order is high.
    
    # Let's check index k which corresponds to "Almost empty".
    # In `Retailer`, K approx 2*delta/mean = 2*150/50 = 6.
    # State k=3 means 3 periods of demand passed. IP approx 150 - 150 = 0.
    # Next demand will trigger order.
    
    # Let's verify State k=3 (approx).
    target_k = 3
    if target_k < len(Dl_list):
        dist = Dl_list[target_k]
        print(f"\nState k={target_k} (Likely low inventory):")
        print(f"  Distribution Mean: {dist.mean()}")
        print(f"  Distribution Std : {dist.std()}")
        
        # Expected behavior:
        # If IP is near 0, next demand (mean 50) triggers order.
        # Order size will be (S - IP_new).
        # Since IP_old ~ 0. IP_new = IP_old - Demand ~ -50.
        # Order = S - (-50) = 200? Or does it cap?
        # Logic: Order = S - IP if IP <= s.
        
        # If order happens, it's roughly sum of (k+1) demands.
        # k=3 -> 3 past demands. + 1 current.
        # Order size ~ (3+1)*50 = 200.
        # Probability of order? High.
        
        # This confirms logic is "Retailer Order Distribution".
        
    else:
        print(f"State k={target_k} out of range.")
        
    # Check k=0
    dist_0 = Dl_list[0]
    print(f"\nState k=0 (Full inventory):")
    print(f"  Mean: {dist_0.mean()} (Should be ~0)")
    
    print("\nTest Finished.")
    
if __name__ == "__main__":
    test_sampling_vs_theory_L0()
