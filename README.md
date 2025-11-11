About this Competition
This competition challenges you to predict the daily returns of the S&P 500 index using a tailored set of market data features.

Competition Phases and Data Updates
The competition will proceed in two phases:

A model training phase with a test set of six months of historical data. Because these prices are publicly available leaderboard scores during this phase are not meaningful.
A forecasting phase with a test set to be collected after submissions close. You should expect the scored portion of the test set to be about the same size as the scored portion of the test set in the first phase.
During the forecasting phase the evaluation API will serve test data from the beginning of the public set to the end of the private set. This includes trading days before the submission deadline, which will not be scored. The first date_id served by the API will remain constant throughout the competition.

Files
train.csv Historic market data. The coverage stretches back decades; expect to see extensive missing values early on.

date_id - An identifier for a single trading day.
M* - Market Dynamics/Technical features.
E* - Macro Economic features.
I* - Interest Rate features.
P* - Price/Valuation features.
V* - Volatility features.
S* - Sentiment features.
MOM* - Momentum features.
D* - Dummy/Binary features.
forward_returns - The returns from buying the S&P 500 and selling it a day later. Train set only.
risk_free_rate - The federal funds rate. Train set only.
market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4. Train set only.
test.csv A mock test set representing the structure of the unseen test set. The test set used for the public leaderboard set is a copy of the last 180 date IDs in the train set. As a result, the public leaderboard scores are not meaningful. The unseen copy of this file served by the evaluation API may be updated during the model training phase.

date_id
[feature_name] - The feature columns are the same as in train.csv.
is_scored - Whether this row is included in the evaluation metric calculation. During the model training phase this will be true for the first 180 rows only. Test set only.
lagged_forward_returns - The returns from buying the S&P 500 and selling it a day later, provided with a lag of one day.
lagged_risk_free_rate - The federal funds rate, provided with a lag of one day.
lagged_market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4, provided with a lag of one day.
