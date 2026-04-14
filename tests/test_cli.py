import unittest
from pathlib import Path
from unittest.mock import patch

from upsampler import cli


class TestCli(unittest.TestCase):
    def test_parse_args(self):
        args = cli.parse_args(["image.png", "--factor", "3", "--output", "out.png"])
        self.assertEqual(args.image, "image.png")
        self.assertEqual(args.factor, 3)
        self.assertEqual(args.output, "out.png")
        self.assertEqual(args.model_id, cli.DEFAULT_MODEL_ID)

    def test_main_invokes_upsample_image(self):
        with patch("upsampler.cli.upsample_image") as mock_upsample:
            rc = cli.main(["image.png", "--factor", "2.5", "--output", "out.png"])

        self.assertEqual(rc, 0)
        mock_upsample.assert_called_once_with(
            Path("image.png"),
            Path("out.png"),
            2.5,
            cli.DEFAULT_MODEL_ID,
        )


if __name__ == "__main__":
    unittest.main()
