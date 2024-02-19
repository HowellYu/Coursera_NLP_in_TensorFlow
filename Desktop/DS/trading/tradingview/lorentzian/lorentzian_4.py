from backtrader import Observer
import backtrader as bt
import numpy as np
from backtrader import Strategy
from backtrader.indicators import ExponentialMovingAverage, SimpleMovingAverage, RelativeStrengthIndex, CCI, ADX


class NormalizedRSI(bt.Indicator):
    lines = ('n_rsi',)
    params = (
        ('n1', 14),
        ('n2', 14),
    )

    def __init__(self):
        rsi = bt.indicators.RSI(period=self.p.n1)
        ema_rsi = bt.indicators.EMA(rsi, period=self.p.n2)
        self.lines.n_rsi = self.rescale(ema_rsi, 0, 100, 0, 1)

    def rescale(self, data, old_min, old_max, new_min, new_max):
        return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class WaveTrend(bt.Indicator):
    lines = ('n_wt',)
    params = (
        ('n1', 10),
        ('n2', 21),  # Adjusted to ensure enough data points for normalization
    )

    def __init__(self):
        ema1 = bt.indicators.EMA(self.data, period=self.p.n1)
        ema2 = bt.indicators.EMA(abs(self.data - ema1), period=self.p.n1)
        ci = (self.data - ema1) / (0.015 * ema2)
        wt1 = bt.indicators.EMA(ci, period=self.p.n2)
        wt2 = bt.indicators.SMA(wt1, period=4)
        # Normalization will be handled dynamically in next
        self.wt1_minus_wt2 = wt1 - wt2

    def next(self):
        # Dynamically calculate min and max for normalization
        window_size = self.p.n2
        if len(self) >= window_size:
            data_min = min(self.wt1_minus_wt2.get(size=window_size))
            data_max = max(self.wt1_minus_wt2.get(size=window_size))
            if data_max - data_min == 0:
                self.lines.n_wt[0] = 0  # Avoid division by zero
            else:
                # Perform normalization
                normalized_value = (
                    self.wt1_minus_wt2[0] - data_min) / (data_max - data_min)
                self.lines.n_wt[0] = normalized_value


class NormalizedCCI(bt.Indicator):
    lines = ('n_cci',)
    params = (
        ('n1', 20),  # Period for CCI
        ('n2', 20),  # Period for EMA applied to CCI
    )

    def __init__(self):
        # Compute the CCI indicator
        cci = bt.indicators.CCI(period=self.p.n1)
        # Compute the EMA of the CCI
        self.ema_cci = bt.indicators.EMA(cci, period=self.p.n2)
        # To keep track of min and max values for normalization
        self.cci_min = float('inf')
        self.cci_max = float('-inf')

    def next(self):
        # Update min and max based on current EMA of CCI value
        self.cci_min = min(self.cci_min, self.ema_cci[0])
        self.cci_max = max(self.cci_max, self.ema_cci[0])

        # Ensure we have a range to normalize within
        if self.cci_max > self.cci_min:
            # Perform normalization
            normalized_cci = (
                self.ema_cci[0] - self.cci_min) / (self.cci_max - self.cci_min)
            self.lines.n_cci[0] = normalized_cci
        else:
            # Default to 0.5 (mid-point) in case of no variation
            self.lines.n_cci[0] = 0.5


class NormalizedADX(bt.Indicator):
    lines = ('n_adx',)
    params = (
        ('n', 14),
    )

    def __init__(self):
        adx = bt.indicators.ADX(period=self.p.n)
        self.lines.n_adx = self.rescale(adx, 0, 100, 0, 1)

    def rescale(self, data, old_min, old_max, new_min, new_max):
        # Similar rescale logic as in NormalizedRSI
        return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class LorentzianKNNStrategy(bt.Strategy):
    params = (
        # Parameters for the strategy
        ('trade_value', 5000),
        ('k', 8),  # Number of neighbors for KNN
        ('feature_count', 5),  # top features to include

        ('rsi_featureA', 14),
        ('rsi_featureB', 14),
        ('wt_featureA', 10),
        ('wt_featureB', 11),
        ('cci_featureA', 20),
        ('cci_featureB', 20),
        ('adx_featureA', 14),
        ('rsi2_featureA', 14),
        ('rsi2_featureB', 14),

        # Parameters for volatility filter
        ('atr_minLength', 1),
        ('atr_maxLength', 10),
        ('useVolatilityFilter', True),

        # Parameters for regime filter
        ('regime_threshold', -0.1),
        ('useRegimeFilter', True),
        ("regime_value1_period", 10),
        ('regime_value2_period', 10),

        # Placeholder for ADX filter
        ('adx_length', 14),
        ('adxThreshold', 25),
        ('useAdxFilter', True),
    )

    def __init__(self):
        # Placeholder for indicators
        # self.ema = ExponentialMovingAverage(period=self.params.ema_period)
        # self.sma = SimpleMovingAverage(period=self.params.sma_period)

        self.data_length = len(list(self.data.close))
        print('Initializing strategy')
        print(f"Backtesting data length: {self.data_length}")

        print('Initializing strategy')

        self.feature1 = NormalizedRSI(
            self.data.close, n1=self.p.rsi_featureA, n2=self.p.rsi_featureB)
        self.feature2 = WaveTrend(  # TODO: check
            self.data, n1=self.p.wt_featureA, n2=self.p.wt_featureB)
        self.feature3 = NormalizedCCI(
            self.data, n1=self.p.cci_featureA, n2=self.p.cci_featureB)
        self.feature4 = NormalizedADX(
            self.data, self.data.low, self.data.close, n=self.p.adx_featureA)
        self.feature5 = NormalizedRSI(
            self.data.close, n1=self.p.rsi2_featureA, n2=self.p.rsi2_featureB)

        # ATR indicators for volatility filter
        self.recentAtr = bt.indicators.AverageTrueRange(
            period=self.p.atr_minLength)
        self.historicalAtr = bt.indicators.AverageTrueRange(
            period=self.p.atr_maxLength)

        # regime filter
        self.ohlc4 = (self.data.high + self.data.low +
                      self.data.close + self.data.open) / 4.0
        try:
            self.delta_ohlc4 = self.ohlc4 - self.ohlc4[-1]
        except:
            self.delta_ohlc4 = 0
        self.regime_value1 = bt.indicators.ExponentialMovingAverage(
            self.delta_ohlc4, period=self.p.regime_value1_period)
        self.market_range = self.data.high - self.data.low
        self.regime_value2 = bt.indicators.ExponentialMovingAverage(
            self.market_range, period=self.p.regime_value2_period)

        # adx filter
        self.adx = bt.indicators.ADX(period=self.p.adx_length)

        # Historical feature data (for demonstration, fill with actual historical data)
        # List of lists, where each sublist represents historical features at time i
        self.historical_arrays = []
        print('initialization finished')

    def start(self):
        # Log when entering start()
        print('Entering start()')

        # Call the original start() method
        super().start()

        # Log when exiting start()
        print('Exiting start()')

    def next(self):
        print('In next')
        # Collect current features
        # Extend this list with other current feature values
        current_features = [self.feature1[0], self.feature2[0],
                            self.feature3[0], self.feature4[0], self.feature5[0]]

        # Make sure there's enough historical data
        if len(self.historical_arrays) > self.p.k:
            # Initialize variables
            distances = []  # To store Lorentzian distances
            predictions = []  # To store predictions from k-nearest neighbors
            lastDistance = -1.0

            # Iterate through historical data to calculate distances and find neighbors
            for i in range(len(self.feature_arrays)):
                if i % 4 == 0:  # Perform calculations every 4 bars
                    d = self.get_lorentzian_distance(
                        i, self.params.feature_count, self.feature_series, self.feature_arrays)
                    if d >= lastDistance:
                        lastDistance = d
                        distances.append(d)
                        # Assuming get_prediction returns a prediction for the i-th historical data point
                        predictions.append(self.get_prediction(i))

                        # Ensure we only keep the k-nearest neighbors
                        if len(predictions) > self.params.neighbors_count:
                            # Keep the last 25% distances as the new threshold
                            lastDistance = sorted(distances)[
                                int(len(distances) * 0.75)]
                            # Remove the oldest neighbors beyond k count
                            distances = distances[-self.params.neighbors_count:]
                            predictions = predictions[-self.params.neighbors_count:]

            # Aggregate predictions from the k-nearest neighbors
            # Placeholder for aggregation logic, here we simply average the predictions
            knn_signal = np.sign(sum(predictions) /
                                 len(predictions)) if predictions else 0

            # regime signal
            omega = abs(self.regime_value1[0] / self.regime_value2[0])
            alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8
            # Simplified KLMF calculation
            self.klmf = alpha * self.ohlc4[0] + (1 - alpha) * self.ohlc4[-1]
            abs_curve_slope = abs(self.klmf - self.ohlc4[-1])
            exponential_avg_abs_curve_slope = bt.indicators.ExponentialMovingAverage(
                abs_curve_slope, period=200)
            normalized_slope_decline = (
                abs_curve_slope - exponential_avg_abs_curve_slope[0]) / max(exponential_avg_abs_curve_slope[0], 1e-5)
            regime_signal = normalized_slope_decline >= self.p.regime_threshold

            if self.position:
                # Check if it's been 'exit_bars' since buying
                if len(self) - self.buy_bar >= self.p.exit_bars:
                    # Exit all positions
                    self.close()

            # Execute trading logic based on prediction
            else:
                if knn_signal > 0 and self.filter_volatility() and regime_signal and self.filter_adx():
                    if not self.position:
                        self.buy(size=self.p.trade_value / self.data.close[0])
                # elif knn_signal < 0 and self.filter_volatility() and not regime_signal and not self.filter_adx():
                #     if self.position:
                #         self.close()

        # At the end of each day, update feature_arrays with current day's features
        self.historical_arrays.append(current_features)

    def get_lorentzian_distance(self, i, feature_count, feature_series, feature_arrays):
        distance = 0
        feature_count = min(feature_count, len(feature_series))
        for j in range(feature_count):
            distance += np.log(1 +
                               abs(feature_series[j] - feature_arrays[i][j]))
        return distance

    def filter_volatility(self):
        """
        Compares recent ATR to historical ATR to determine if volatility has increased.

        Returns:
        - True if recent ATR is greater than historical ATR (and using volatility filter),
          or if the volatility filter is not used.
        - False otherwise.
        """
        if self.p.useVolatilityFilter:
            return self.recentAtr[0] > self.historicalAtr[0]
        return True

    def regime_filter(self):
        # Calculate omega and alpha based on regime_value1 and regime_value2
        omega = abs(self.regime_value1[0] / self.regime_value2[0])
        alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8

        # KLMF calculation with alpha
        # Approximation for klmf using ohlc4
        klmf = alpha * self.ohlc4[0] + (1 - alpha) * self.ohlc4[-1]

        # Calculate the absolute curve slope and its exponential moving average
        abs_curve_slope = abs(klmf - self.ohlc4[-1])
        exponential_avg_abs_curve_slope = bt.indicators.ExponentialMovingAverage(
            abs_curve_slope, period=200)

        # Normalized slope decline
        normalized_slope_decline = (
            abs_curve_slope - exponential_avg_abs_curve_slope[0]) / max(exponential_avg_abs_curve_slope[0], 1e-5)

        # Apply the regime filter logic
        return self.p.useRegimeFilter and normalized_slope_decline >= self.p.regime_threshold

    def filter_adx(self):
        """
        Checks if the ADX value is above a specified threshold.

        Returns:
        - True if ADX > adxThreshold and useAdxFilter is True.
        - True if useAdxFilter is False (ignoring ADX filter condition).
        - False otherwise.
        """
        if self.p.useAdxFilter:
            return self.adx[0] > self.p.adxThreshold
        return True
# Additional code for setting up data, Cerebro engine, etc., is needed to run this strategy.

# Additional setup for Backtrader (data feed, Cerebro, etc.) is required to execute this strategy.


class ProgressObserver(Observer):
    alias = ('ProgressObserver',)
    lines = ('progress',)

    def __init__(self):
        self.data_length = len(list(self.data.close))
        print('Initializing observer')
        print(
            f"Backtesting data length: {self.data_length}")
        self.report_interval = self.data_length // 10
        print('Observer initialized')

    def next(self):
        current_index = len(self.datas[0])
        print(f"Backtesting progress: {current_index}/{self.data_length}")
        if current_index % self.report_interval == 0:
            progress_percentage = (current_index / self.data_length) * 100
            print(f"Backtesting progress: {progress_percentage:.2f}%")
