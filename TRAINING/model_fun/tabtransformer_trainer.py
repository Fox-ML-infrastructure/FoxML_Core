# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

from .transformer_trainer import TransformerTrainer as TabTransformerTrainer

# TabTransformer is an alias for Transformer - they use the same architecture
# The enhanced TransformerTrainer already has all the safety features