# Public Libraries
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm
from line_profiler import profile
import warnings
import traceback
from scipy import stats

warnings.filterwarnings('ignore', message='All-NaN slice encountered')

# Private Libraries
from Analysis.utilities import * 

# Update Log
# 2024-01-28 (v4): zscore -> t-test, include robust holding period testing
# So far best strategy resulted from:
    # using v2, zscore method with 3.271, 
    # final score latest 30 zscore
    # then filtering once more for robust zscore at 2.


##################################################################################################################
##################################################################################################################
class Target ():
    def __init__ (self, dir_path, file, start_date = None, end_date = None):
        
        # Load Target OHLCV
        self.ohlcv = pd.read_csv(os.path.join(dir_path, file))
        self.ohlcv.Time = pd.to_datetime(self.ohlcv.Time, utc=True)
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True)            
        
        self.file = file
        self.end_date = end_date
        self.symbol = file.strip().split('_')[0]
        self.freq  = self.ohlcv.Time.diff().mode()[0].seconds//60
        total_days = (self.ohlcv.Time.iloc[-1]-self.ohlcv.Time.iloc[0]).days

        # Test Parameters
        self.test_params = {
            'min_data': 569,
            'min_days': 60,
            'max_overlap': 0.20,
            'max_pct_passed': 0.90,
            'p_value' : 0.001,
            'robust_p_value': 0.05,
            'min_cumlret': 0,
            'min_avg_lret': 0,
            'min_vol': 0,
            'min_vol_lookback': 30, 
            'min_lret_per_24h': 0,
            'n_periods': 4,
            'macro_scales': [3,12,48],
            'min_group_size': 0,
            'B_params': {'lookback':30}
        }

        # Pre-Compute: Return Matrix
        hold_periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50]
        self.returns_matrix, self.return_header = create_return_matrix(self.ohlcv, hold_periods, commission = 0.001)

        hold_periods_short = [1, 1, 2, 2, 3, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
        self.returns_shortened, _ = create_return_matrix(self.ohlcv, hold_periods_short, commission = 0.001)

        hold_periods_long = [2, 3, 4, 6, 7, 9, 12, 15, 18, 24, 30, 36, 45, 60, 75]
        self.returns_extended, _ = create_return_matrix(self.ohlcv, hold_periods_long, commission = 0.001)

        # Pre-Compute: Time Matrix
        self.Time_bool, self.Time_header = create_time_signal_matrix(self.ohlcv.Time, min_data = self.test_params['min_data'])

        # Pre-Compute: Misc.
        self.ohlcv_index = self.ohlcv.index.values
        self.n_candles_per_day = 24*60/self.freq
        self.max_day_index = self.ohlcv.index[-1]//self.n_candles_per_day
        self.subperiods = get_subperiod_indices(self.max_day_index, self.test_params['n_periods'])
        # self.latest_sharp_day_index = self.max_day_index - (self.test_params['latest_sharpe_months']*30)

##################################################################################################################

    def run (self):

        ### List of Features to Test ###
        feature1 = {
            'A_name': "bollinger",
            'A_func': bollinger,
            'A_data': self.ohlcv,
            'A_params': {'lookback':30},
            'A_thresh_transf': "as_is",
            'A_thresh_val': [2,3,4,5,6,7,8,10,12,14, 16, -2,-3,-4,-5,-6,-8, -10, -12, -14, -16],
            'A_thresh_method': "crossing_2way",
        }
        
        feature2 = {
            'A_name': "sigfig_cross",
            'A_func': sigfig_cross,
            'A_data': self.ohlcv,
            'A_params': {'n_sigfig': 2},
            'A_thresh_transf': "as_is",
            'A_thresh_val': [-1,1],
            'A_thresh_method': "crossing_1way",
        }

        feature3 = {
            'A_name': "sigfig_cross",
            'A_func': sigfig_cross,
            'A_data': self.ohlcv,
            'A_params': {'n_sigfig': 1},
            'A_thresh_transf': "as_is",
            'A_thresh_val': [-1,1],
            'A_thresh_method': "crossing_1way",
        }

        # feature4 = {
        #     'A_name': "bollinger_vwap",
        #     'A_func': bollinger_vwap,
        #     'A_data': self.ohlcv,
        #     'A_params': {'lookback': 60},
        #     'A_thresh_transf': "as_is",
        #     'A_thresh_val': [1,2,3,4,5,6,7,-1,-2,-3,-4,-5,-6,-7],
        #     'A_thresh_method': "crossing_2way", 
        # }



        df_all, df_select = pd.DataFrame(), pd.DataFrame()
        feature_list = [feature1, feature2, feature3]

        for feature in feature_list:
            
            # Test the signal
            result = self.test_signal(feature)

            # Refine the result
            all_signals, select_signals  = self.select_signal(feature, result)

            # Save it
            df_all = pd.concat([df_all, all_signals], axis=0)
            df_select = pd.concat([df_select, select_signals], axis=0)

        return df_all, df_select

##################################################################################################################

    def signal_generator(self, *, A_name=None, A_func=None, A_data=None, A_params=None, A_thresh_transf=None, A_thresh_val=None, A_thresh_method=None):
        
        min_data = self.test_params['min_data']

        A_transf = A_func(A_data, params = A_params)
        A_thresh = create_thresholds(A_transf, A_thresh_transf, A_thresh_val)
        A_bool, A_header = create_signal_matrix(A_transf, A_thresh,A_thresh_method, min_data, include_all = False)
        C_bool, C_header = self.Time_bool, self.Time_header

        # Apply minimum volatility filter
        atr = calc_atr(A_data, self.test_params['min_vol_lookback'])
        atr_quantile = atr.rolling(self.test_params['min_vol_lookback']*100).rank()/(self.test_params['min_vol_lookback']*100)        
        below_min_vol = atr_quantile<self.test_params['min_vol']
        A_bool[below_min_vol,:] = False

        # Apply Date Filter
        if self.start_date is not None:
            start_filter = (A_data.Time < self.start_date).values
            A_bool[start_filter,:] = False
        if self.end_date is not None:
            end_filter = (A_data.Time > self.end_date).values
            A_bool[end_filter,:] = False


        for macro_scale in self.test_params['macro_scales']:
            
            B_transf = macro_bollinger(A_data, macro_scale, params = self.test_params['B_params'])
            B_thresh = create_thresholds(B_transf, "as_is", [-10,-8,-6,-4,-2,0,2,4,6,8,10])
            B_bool, B_header = create_signal_matrix(B_transf, B_thresh, "macro_mode_zscore", min_data, include_all = True)

            # Iterate through our conditions
            for i in range(A_bool.shape[1]):
                for j in range(B_bool.shape[1]): # We can further vectorize this into 4D
                    
                    # Combine A & B
                    AB_1d = A_bool[:,i] & B_bool[:,j]

                    # The idea is if B filters out only 1 or 2, then it could be an overfit. 
                    pct_passed = AB_1d.sum()/A_bool[:,i].sum()
                    if (AB_1d.sum() < min_data) or (pct_passed > self.test_params['max_pct_passed']): 
                        continue
                    
                    # Combine AB & C
                    for k in range(C_bool.shape[1]):

                        signal = AB_1d & C_bool[:,k]                        
                        pct_passed = signal.sum()/AB_1d.sum()
                        if (signal.sum() < min_data) or (pct_passed > self.test_params['max_pct_passed']): 
                            continue

                        signal_info = np.concatenate((np.array([self.freq]), A_header[:,i], np.array([macro_scale]), B_header[:,j], C_header[:,k]))
                        yield signal, signal_info

##################################################################################################################

    def test_signal(self, feature):

        final_result = []
        min_data = self.test_params['min_data']

        # Iterate through each signal
        for signal, signal_info in self.signal_generator(**feature):

            # Build matrix of entry indices across exit configurations (=columns)
            signal_index_1d = self.ohlcv_index[signal]
            signal_index_wOverlap_2d = (signal_index_1d[:, np.newaxis] * np.ones(self.returns_matrix.shape[1])).astype('int')

            # Fetch returns from the matrix
            signal_lrets_2d = self.returns_matrix[signal_index_wOverlap_2d, np.arange(signal_index_wOverlap_2d.shape[1])]
            
            # Filter #1
            filter1 = np.sum(signal_lrets_2d, axis=0) > self.test_params['min_cumlret']
            if filter1.sum() == 0 : 
                continue

            # Update/Filter numbers
            current_params_2d = self.return_header[:,filter1]
            current_signal_index_2d = signal_index_wOverlap_2d[:, filter1]
            current_lrets_2d = signal_lrets_2d[:,filter1]

            # Processing: Remove Overlaps, Convert to daily lrets
            current_signal_index_2d, overlap_mask = remove_overlapping_signals(current_signal_index_2d, distance = current_params_2d[0,:], fill_value = -1)
            current_lrets_2d = current_lrets_2d*~overlap_mask
            signal_daily_lrets_2d, _ = resample_into_daily(current_lrets_2d, self.freq, signal_index_1d)

            if len(signal_daily_lrets_2d) < self.test_params['min_days']:
                continue

            # Check: Minimum Data Count
            signal_count = np.sum(current_signal_index_2d>0, axis=0)
            min_data_filter = signal_count > min_data

            # Check: Excessive Overlap
            signal_count_wOverlap = np.sum(signal_index_wOverlap_2d[:,filter1]>0, axis=0)
            overlap_filter = (1 - signal_count/signal_count_wOverlap) < self.test_params['max_overlap']

            # Check: P-value
            p_values = t_test_2d(signal_daily_lrets_2d)
            p_value_filter = p_values <= self.test_params['p_value']

            # Filter #2
            filter2 = min_data_filter&overlap_filter&p_value_filter
            if filter2.sum() == 0:
                continue            
            
            # Update/Filter numbers
            current_params_2d = current_params_2d[:,filter2]
            current_lrets_2d = current_lrets_2d[:,filter2]  
            current_signal_index_2d = current_signal_index_2d[:,filter2]
            current_daily_lrets = signal_daily_lrets_2d[:,filter2]
            current_count = signal_count[filter2]
            current_p_values = p_values[filter2]

            current_lrets_2d_masked = np.ma.masked_array(current_lrets_2d, mask=(current_lrets_2d == 0))
            avg_lrets =  np.ma.mean(current_lrets_2d_masked, axis=0)
            lret_per_24h = avg_lrets / current_params_2d[0,:] * (24*60/self.freq)

            # Robustness Testing
            returns_shortened = self.returns_shortened[:,filter1][:,filter2]
            returns_extended = self.returns_extended[:,filter1][:,filter2]
            signal_lrets_shortened_2d = returns_shortened[current_signal_index_2d, np.arange(current_signal_index_2d.shape[1])]            
            signal_lrets_extended_2d = returns_extended[current_signal_index_2d, np.arange(current_signal_index_2d.shape[1])]
            signal_daily_lrets_shortened_2d, _ = resample_into_daily(signal_lrets_shortened_2d, self.freq, signal_index_1d)
            signal_daily_lrets_extended_2d, _ = resample_into_daily(signal_lrets_extended_2d, self.freq, signal_index_1d)
            p_values_shortened = t_test_2d(signal_daily_lrets_shortened_2d)
            p_values_extended = t_test_2d(signal_daily_lrets_extended_2d)
            filter3 = (p_values_shortened <= self.test_params['robust_p_value']) & (p_values_extended <= self.test_params['robust_p_value'])
            if filter3.sum() == 0:
                continue

            # Update/Filter numbers
            current_params_2d = current_params_2d[:,filter3]
            current_lrets_2d = current_lrets_2d[:,filter3]  
            current_signal_index_2d = current_signal_index_2d[:,filter3]
            current_daily_lrets = current_daily_lrets[:,filter3]
            current_count = current_count[filter3]
            current_p_values = current_p_values[filter3]

            current_p_values_shortened = p_values_shortened[filter3]
            current_p_values_extended = p_values_extended[filter3]

            current_avg_lrets = avg_lrets[filter3]
            current_lret_per_24h = lret_per_24h[filter3]

            # Check latest zscore
            last_30_daily_lrets = current_daily_lrets[-30:,:]
            latest_p_value = t_test_2d(last_30_daily_lrets)
            # latest_zscore_filter = latest_zscores < self.test_params['min_latest_zscore']

            # # Assuming 'current_daily_lrets' is your original 2D array
            # sub_zscores = []  # List to store z-scores for each block

            # Calculate z-scores for every block of 30 elements, skipping blocks with less than 20 elements
            # n = len(current_daily_lrets)
            # if n < 25:
            #     continue
            # for end_idx in range(n, 0, -30):
            #     start_idx = max(end_idx - 30, 0)  # Ensure start index doesn't go below 0

            #     # Skip the block if it has less than 20 elements
            #     if end_idx - start_idx < 15:
            #         continue

            #     block = current_daily_lrets[start_idx:end_idx, :]
            #     zscores = calc_zscore_2d(block)
            #     sub_zscores.append(zscores)
            # sub_zscores = np.vstack(sub_zscores)
            # if sub_zscores.shape[0] < 7:
            #     rows_to_add = 7 - sub_zscores.shape[0]
            #     sub_zscores = np.vstack((sub_zscores, np.full((rows_to_add, sub_zscores.shape[1]), np.nan)))

            # Headers
            n_col = current_params_2d.shape[1]
            signal_info = np.repeat(signal_info.reshape(-1,1),n_col,axis=1)


            # Save Result
            result = np.vstack((
                signal_info, 
                current_params_2d, current_p_values, current_p_values_shortened, current_p_values_extended, latest_p_value, current_count, \
                    current_avg_lrets, current_lret_per_24h, 
                )).T
            
            final_result.append(result)

        return final_result

##################################################################################################################

    def select_signal(self, feature, result):
        if len(result) == 0:
            return None, None
        column_names = ['freq','A_thresh','A_dir','B_scale','B_thresh','B_dir','timezone','dayofweek','timeofday','hold_periods','gap','side','p_value','p_value_short','p_value_long', 'latest_p_value','count', 'avg_lrets', 'lret_per_24h']
        df_all_signals = pd.DataFrame(np.vstack(result), columns = column_names)

        # Calculate Signal Score
        final_score = 1/df_all_signals['latest_p_value']
        # sharpe_score = np.minimum(4,df_all_signals['min_sharpe'])
        # return_score = np.minimum(10, df_all_signals['avg_lret']/0.01)
        # efficiency_score = np.minimum(10,((df_all_signals['lret_per_24h']/0.1)**0.5))
        # test_score = df_all_signals['avg_lret']/abs(df_all_signals['path_low'])

        # df_all_signals['final_score'] = (df_all_signals['latest_zscore']).astype(float)
        df_all_signals['final_score'] = final_score.astype(float)

        # Add Feature Information
        df_all_signals.insert(0,'file', self.file)
        df_all_signals.insert(1,'end_date', self.end_date)
        df_all_signals.insert(2, 'symbol', self.symbol)
        df_all_signals.insert(3, 'feature_name', feature['A_name'])
        df_all_signals.insert(4, 'feature_param', str(feature['A_params']))
        df_all_signals.insert(5, 'min_vol_quantile', self.test_params['min_vol'])

        # Group them by 
        grouped = df_all_signals.groupby(['A_thresh', 'side'])        
        df_all_signals['group_size'] = grouped['final_score'].transform('size')

        # Select the best signal for each group (unique combo of threshold and side)
        # Use group size to filter out potential overfits
        df_filtered = df_all_signals[(df_all_signals['group_size'] >= self.test_params['min_group_size'])]
        grouped_filtered = df_filtered.groupby(['A_thresh', 'side'])
        max_indices = grouped_filtered['final_score'].idxmax()
        df_select_signals = df_filtered.loc[max_indices] if len(max_indices) > 0 else None

        return df_all_signals, df_select_signals

##################################################################################################################
##################################################################################################################

## Execution ##

def process_file(dir_path, file, start_date = None, end_date = None):
    target = Target(dir_path, file, start_date, end_date)
    output = target.run()
    return output

def test_one_file():
    df_all_filepath = "results/test_all.csv" 
    df_select_filepath =  "results/test_select.csv" 
    dir_path = "data/2yrs_231231"
    file = "1000SHIBUSDT_5m_210829_231231.csv"

    end_date = None #pd.to_datetime("2023-07-31", utc=True)

    df_all, df_select = process_file(dir_path,file, end_date = end_date)
    df_all.to_csv(df_all_filepath)
    df_select.to_csv(df_select_filepath)

    # pd.set_option('display.max_columns', None)
    # print(df_all)
    # print(df_select)
    
def process_all_files_in_parallel(num_cpus=8):
    
    # Output Path
    df_all_filepath = "results/trained_from221120_to231130_p1_robust5_all.csv" 
    df_select_filepath =  "results/trained_from221120_to231130_p1_robust5_select.csv" 
    
    # Input Path
    dir_path = "data/2yrs_231231"

    # Input Data Period 
    start_date = "2022-11-20"
    end_date = "2023-11-30"

    df_all = pd.DataFrame()
    df_select = pd.DataFrame()

    # Execute in parallel
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

    with ProcessPoolExecutor(max_workers= num_cpus) as executor:
        # Use the submit method to get futures for each file processing task
        future_to_file = {executor.submit(process_file, dir_path, file, start_date, end_date): file for file in files}

        # Use tqdm with as_completed to track which futures complete in real-time and update the progress bar accordingly
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing files"):
            file = future_to_file[future]
            try:
                # Get results
                all_signals, select_signals = future.result()
                df_select = pd.concat([df_select, select_signals], axis=0, ignore_index=True)
                df_all = pd.concat([df_all, all_signals], axis=0, ignore_index=True)

                # Save as we go
                print(f'{file} done: {len(select_signals)} added.')
                df_select.to_csv(df_select_filepath)
                df_all.to_csv(df_all_filepath)
            
            except Exception as e:
                print(f'\n{file} generated an exception: {e.__class__.__name__}')
                print(f'\nArguments:', e.args)
                print('\nDetailed traceback:')
                traceback.print_tb(e.__traceback__)

##################################################################################################################                
##################################################################################################################

def main():
    process_all_files_in_parallel()
    # test_one_file()    

if __name__ == '__main__':
    try: 
        main()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
