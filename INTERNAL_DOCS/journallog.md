# Watch target ranking in real-time
journalctl -f SYSLOG_IDENTIFIER=rank_target_predictability

# Watch feature ranking
journalctl -f SYSLOG_IDENTIFIER=rank_features_by_ic_and_predictive

# Watch all ranking scripts
journalctl -f | grep -E "SYSLOG_IDENTIFIER=(rank_target|rank_features|multi_model)"

# Last 100 lines
journalctl -n 100 SYSLOG_IDENTIFIER=rank_target_predictability

# Filter by level (warnings/errors only)
journalctl -p warning SYSLOG_IDENTIFIER=rank_target_predictability
