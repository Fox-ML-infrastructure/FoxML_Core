"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# scripts/common/tickers.py
import re
import logging

logger = logging.getLogger(__name__)

CLASS_FIX = {
    r'(?i)^BRK[/\.-]B$': 'BRK.B',
    r'(?i)^BRK[/\.-]A$': 'BRK.A',
    r'(?i)^BF[/\.-]B$': 'BF.B',
    r'(?i)^HEI[/\.-]A$': 'HEI.A',
    r'(?i)^LEN[/\.-]B$': 'LEN.B',
}

# CAREFUL: Only blacklist symbols you actually can't trade
# Removing SQ, GOLD, X as they are tradable - only add if truly non-tradable
CONFIRMED_BLACKLIST = {
    # Add only symbols that are genuinely non-tradable on Alpaca
    # "DELISTED_SYMBOL",  # Example - only use if actually delisted
}

def normalize_symbol(s: str) -> str:
    """Normalize symbol to broker-compatible format."""
    s = s.strip().upper().replace('/', '.').replace('-', '.')
    for pat, rep in CLASS_FIX.items():
        if re.match(pat, s): 
            return rep
    return s

def audit_blacklist(blk):
    """Audit blacklist contents for transparency."""
    items = sorted(blk)
    logger.info(f"🔒 Blacklist loaded ({len(items)}): {items}")

def is_blacklisted(sym: str, blk = None) -> tuple[bool, str]:
    """Check if symbol is blacklisted (exact matches only to avoid accidents)."""
    if blk is None:
        blk = CONFIRMED_BLACKLIST
    # Require exact matches only - avoid substring/regex accidents
    if sym in blk:
        return (False, f"Symbol {sym} is blacklisted (exact match)")
    return (True, "")

def audit_normalizations(symbols: list) -> dict:
    """
    Audit symbol normalizations for transparency.
    Returns dict with normalization stats and changes.
    """
    changes = []
    blacklisted = []
    
    for sym in symbols:
        normalized = normalize_symbol(sym)
        allowed, reason = is_blacklisted(normalized)
        
        if sym != normalized:
            changes.append((sym, normalized))
            
        if not allowed:
            blacklisted.append((sym, reason))
    
    return {
        "total_symbols": len(symbols),
        "normalizations": changes,
        "blacklisted": blacklisted,
        "pass_through": len(symbols) - len(changes) - len(blacklisted)
    }

def process_symbol_universe(raw_symbols: list) -> tuple[list, dict]:
    """
    Process raw symbol list with normalization and blacklist filtering.
    Returns (clean_symbols, audit_report)
    """
    clean_symbols = []
    audit = audit_normalizations(raw_symbols)
    
    for sym in raw_symbols:
        normalized = normalize_symbol(sym)
        allowed, reason = is_blacklisted(normalized)
        
        if allowed:
            clean_symbols.append(normalized)
        else:
            logger.info(f"Excluding {sym}: {reason}")
    
    logger.info(f"Symbol processing: {len(raw_symbols)} → {len(clean_symbols)} symbols")
    if audit["normalizations"]:
        logger.info(f"Normalizations applied: {dict(audit['normalizations'])}")
    if audit["blacklisted"]:
        logger.info(f"Blacklisted: {[f'{s}: {r}' for s, r in audit['blacklisted']]}")
    
    return clean_symbols, audit
