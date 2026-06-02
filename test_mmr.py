import math
import unittest
from unittest.mock import MagicMock
import sys
import unittest.mock

# Create mock store
sys.modules['sqlite_vec'] = MagicMock()
from vstash.store import VstashStore

class TestMMRLazy(unittest.TestCase):
    def test_mmr_lazy_eval(self):
        # mock VstashStore internal DB methods
        VstashStore._connect = lambda self: MagicMock()
        store = VstashStore(db_path=':memory:')

        ranked = [
            {"id": "1", "path": "/doc1", "rrf": 0.9},
            {"id": "2", "path": "/doc1", "rrf": 0.8},
            {"id": "3", "path": "/doc2", "rrf": 0.7},
        ]

        # mock embedding fetch
        def mock_execute(query, params):
            mock_cursor = MagicMock()
            # return some dummy binary embeddings
            import struct
            mock_cursor.fetchall.return_value = [
                {"rowid": 1, "embedding": struct.pack("2f", 1.0, 0.0)},
                {"rowid": 2, "embedding": struct.pack("2f", 0.0, 1.0)},
            ]
            return mock_cursor

        store._conn.execute = mock_execute

        result = store._mmr_dedup(ranked, top_k=2, mmr_lambda=0.5)
        self.assertEqual(len(result), 2)
        # Verify it didn't crash and preserved ordering logic
        self.assertEqual(result[0]["id"], "1")

if __name__ == '__main__':
    unittest.main()
