def calculate_adx(price_df, window=14):
    adx = price_df.ta.adx(length=window, append=False)
    return adx["ADX_14"]


def generate_adx_signals(adx_df, adx_threshold=25):
    latest_adx = adx_df.iloc[-1]
    adx_trending = latest_adx > adx_threshold
    adx_trending = adx_trending.fillna(False)
    return adx_trending
