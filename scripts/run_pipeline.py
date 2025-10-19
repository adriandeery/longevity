# scripts/run_pipeline.py
"""
Master script to run the entire pipeline from data fetching to hypothesis generation.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from longevity.logging.logger import logger
from longevity.config.settings import settings


def run_command(cmd, description):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {result.stderr}")
        return False

    logger.info(f"Success: {description}")
    if result.stdout:
        logger.info(result.stdout)
    return True


def setup_data_directories():
    """Create necessary data directories."""
    dirs = [
        settings.data_dir,
        settings.data_dir / "pubmed",
        settings.data_dir / "index",
        settings.models_dir,
        Path("logs"),
        Path("outputs"),
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")


def check_environment():
    """Check if environment is properly set up."""
    logger.info("Checking environment...")

    # Check Python version
    if sys.version_info < (3, 10):
        logger.error("Python 3.10+ required")
        return False

    # Check CUDA availability
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        logger.warning("PyTorch not installed")

    # Check scispacy model
    try:
        import spacy

        nlp = spacy.load("en_core_sci_sm")
        logger.info("Scispacy model loaded successfully")
    except Exception as e:
        logger.error(f"Scispacy model not found: {e}")
        logger.info("Run: python -m spacy download en_core_sci_sm")
        return False

    # Check Entrez email
    if not settings.entrez_email:
        logger.error("ENTREZ_EMAIL not set in environment")
        logger.info("Set with: export ENTREZ_EMAIL=your@email.com")
        return False

    logger.info("Environment check passed")
    return True


def run_full_pipeline(args):
    """Run the complete pipeline."""

    logger.info("=" * 80)
    logger.info("STARTING LONGEVITY HYPOTHESIS GENERATION PIPELINE")
    logger.info("=" * 80)

    # Step 0: Setup
    setup_data_directories()

    if not check_environment():
        logger.error("Environment check failed. Please fix issues and try again.")
        return False

    # Step 1: Fetch PubMed data
    if args.fetch_data:
        logger.info("\n[STEP 1/5] Fetching PubMed data")
        cmd = [
            "python",
            "-m",
            "longevity.components.fetch_pubmed",
            "--query",
            args.query or "hallmarks of aging",
            "--n",
            str(args.num_papers),
            "--out",
            str(settings.data_dir / "pubmed" / "raw.jsonl"),
        ]
        if not run_command(cmd, "Fetch PubMed abstracts"):
            return False

    # Step 2: Preprocess data
    if args.preprocess:
        logger.info("\n[STEP 2/5] Preprocessing data")
        cmd = [
            "python",
            "-m",
            "longevity.components.preprocess",
            "--in",
            str(settings.data_dir / "pubmed" / "raw.jsonl"),
            "--out",
            str(settings.data_dir / "pubmed" / "processed.jsonl"),
        ]
        if not run_command(cmd, "Preprocess abstracts"):
            return False

    # Step 3: Build index
    if args.build_index:
        logger.info("\n[STEP 3/5] Building FAISS index")
        cmd = [
            "python",
            "-m",
            "longevity.components.build_index",
            "--in",
            str(settings.data_dir / "pubmed" / "processed.jsonl"),
            "--out",
            str(settings.index_path),
        ]
        if not run_command(cmd, "Build FAISS index"):
            return False

    # Step 4: Generate hypotheses
    if args.generate:
        logger.info("\n[STEP 4/5] Generating hypotheses")
        query = (
            args.hypothesis_query
            or "Can NAD+ supplementation extend healthspan in humans?"
        )

        cmd = [
            "python",
            "-m",
            "longevity.pipeline.inference_pipeline",
            "--query",
            query,
            "--top-k",
            str(args.top_k),
            "--model",
            args.model,
            "--output",
            str(Path("outputs") / f"hypothesis_{Path(query).stem[:30]}.json"),
        ]
        if not run_command(cmd, "Generate hypotheses"):
            return False

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete longevity hypothesis generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline from scratch
  python scripts/run_pipeline.py --full
  
  # Only generate hypotheses (assumes data already indexed)
  python scripts/run_pipeline.py --generate --hypothesis-query "Do senolytics improve cognitive function?"
  
  # Fetch new data and rebuild index
  python scripts/run_pipeline.py --fetch-data --preprocess --build-index --query "rapamycin longevity"
        """,
    )

    # Pipeline stages
    parser.add_argument("--full", action="store_true", help="Run all pipeline stages")
    parser.add_argument("--fetch-data", action="store_true", help="Fetch PubMed data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess data")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index")
    parser.add_argument("--generate", action="store_true", help="Generate hypotheses")

    # Parameters
    parser.add_argument(
        "--query", default="hallmarks of aging", help="PubMed search query"
    )
    parser.add_argument(
        "--num-papers", type=int, default=100, help="Number of papers to fetch"
    )
    parser.add_argument("--hypothesis-query", help="Question for hypothesis generation")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of papers to retrieve"
    )
    parser.add_argument(
        "--model", default="google/flan-t5-base", help="Generation model"
    )

    args = parser.parse_args()

    # If --full, enable all stages
    if args.full:
        args.fetch_data = True
        args.preprocess = True
        args.build_index = True
        args.generate = True

    # Check if at least one stage is selected
    if not any([args.fetch_data, args.preprocess, args.build_index, args.generate]):
        parser.print_help()
        print(
            "\nError: No pipeline stages selected. Use --full or specify individual stages."
        )
        sys.exit(1)

    success = run_full_pipeline(args)
    sys.exit(0 if success else 1)
