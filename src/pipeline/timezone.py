UTC = 'UTC'
LOCAL_TZ = 'America/New_York'
def normalize_to_utc(ts):
    return ts.tz_convert(UTC)
