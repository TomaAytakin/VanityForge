import unittest
from unittest.mock import MagicMock, patch
import json
import io
import sys

# Import the script
import grinder_worker

class TestGrinderWorker(unittest.TestCase):
    @patch('sys.stdin', new_callable=io.StringIO)
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('google.cloud.firestore.Client')
    def test_main(self, mock_firestore_client, mock_stdout, mock_stdin):
        # Setup stdin
        mock_stdin.write(json.dumps({
            "job_id": "test_job_123",
            "prefix": "A",
            "suffix": ""
        }))
        mock_stdin.seek(0)

        # Mock Firestore client and document update
        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_doc_ref = MagicMock()
        mock_db.collection.return_value.document.return_value = mock_doc_ref

        # Run main
        grinder_worker.main()

        # Check if Firestore was updated
        mock_db.collection.assert_called_with("vanity_jobs")
        mock_db.collection.return_value.document.assert_called_with("test_job_123")
        mock_doc_ref.update.assert_called_once()
        args, _ = mock_doc_ref.update.call_args
        update_data = args[0]
        self.assertEqual(update_data['status'], 'COMPLETED')
        self.assertTrue(update_data['public_key'].startswith('A'))

        # Check stdout for secret key array
        output = mock_stdout.getvalue()
        secret_key = json.loads(output)
        self.assertIsInstance(secret_key, list)
        self.assertEqual(len(secret_key), 64)

if __name__ == '__main__':
    unittest.main()
