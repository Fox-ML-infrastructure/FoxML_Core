#!/usr/bin/env python

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

"""List available symbols in the dataset"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

def list_symbols(data_dir: Path = None):
    """List all available symbols in dataset"""

    if data_dir is None:
        data_dir = _REPO_ROOT / "data/data_labeled/interval=5m"
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return []
    
    symbol_dirs = sorted([d for d in data_dir.glob("symbol=*") if d.is_dir()])
    
    symbols = []
    for symbol_dir in symbol_dirs:
        symbol_name = symbol_dir.name.replace("symbol=", "")
        parquet_file = symbol_dir / f"{symbol_name}.parquet"
        if parquet_file.exists():
            symbols.append(symbol_name)
    
    return symbols


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--count", action="store_true", help="Just show count")
    args = parser.parse_args()
    
    symbols = list_symbols(args.data_dir)
    
    if args.count:
        print(f"Total symbols: {len(symbols)}")
    else:
        print(f"Available symbols ({len(symbols)}):")
        print()
        for i, symbol in enumerate(symbols, 1):
            print(f"{i:4d}. {symbol}")
        print()
        print(f"Total: {len(symbols)} symbols")
        print()
        print("Suggested test symbols (5 diverse, liquid):")
        # Pick diverse symbols if available
        suggestions = []
        preferred = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'XOM', 'JNJ', 'WMT', 'V', 'META']
        for sym in preferred:
            if sym in symbols:
                suggestions.append(sym)
            if len(suggestions) >= 5:
                break
        
        if len(suggestions) < 5:
            # Fill with first available
            for sym in symbols[:10]:
                if sym not in suggestions:
                    suggestions.append(sym)
                if len(suggestions) >= 5:
                    break
        
        print(f"  {','.join(suggestions)}")

