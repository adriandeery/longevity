# longevity/pipeline/inference_pipeline.py
"""
End-to-end inference pipeline for longevity hypothesis generation.

Pipeline stages:
1. Query understanding & expansion
2. Literature retrieval (RAG)
3. Hypothesis generation
4. Evidence scoring
5. (Optional) Structure prediction for proposed targets
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

from longevity.components.generate_hypotheses import (
    load_rag_system,
    retrieve_documents,
    generate_hypotheses_structured,
)
from longevity.components.evidence_scoring import (
    score_by_bradford_hill_aging,
    identify_aging_hallmarks,
)
from longevity.config.settings import settings
from longevity.logging.logger import logger
from longevity.exception.exception import LongevityException
import sys


class LongevityInferencePipeline:
    """
    Complete pipeline for generating and scoring longevity hypotheses.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        generation_model: Optional[str] = None,
        use_cuda: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            index_path: Path to FAISS index
            generation_model: Model name for hypothesis generation
            use_cuda: Whether to use CUDA if available
        """
        self.index_path = index_path or str(settings.index_path)
        self.generation_model = generation_model or settings.generation_model
        self.use_cuda = use_cuda

        logger.info("Initializing Longevity Inference Pipeline")
        logger.info(f"Index path: {self.index_path}")
        logger.info(f"Generation model: {self.generation_model}")

        # Load RAG system
        try:
            self.index, self.docs = load_rag_system(self.index_path)
            logger.info(f"Successfully loaded {len(self.docs)} documents")
        except Exception as e:
            raise LongevityException(f"Failed to initialize pipeline: {str(e)}", sys)

    def run(
        self,
        query: str,
        top_k: int = 5,
        score_evidence: bool = True,
        identify_hallmarks: bool = True,
    ) -> Dict:
        """
        Run the complete pipeline.

        Args:
            query: Research question
            top_k: Number of papers to retrieve
            score_evidence: Whether to score evidence using Bradford Hill
            identify_hallmarks: Whether to identify aging hallmarks

        Returns:
            Complete results dictionary
        """
        logger.info(f"Running pipeline for query: {query}")
        start_time = datetime.now()

        try:
            # Stage 1: Generate hypotheses (includes retrieval)
            generation_result = generate_hypotheses_structured(
                query=query,
                index_path=self.index_path,
                model_name=self.generation_model,
                k=top_k,
            )

            # Stage 2: Score evidence
            evidence_scores = None
            if score_evidence:
                logger.info("Scoring evidence with Bradford Hill criteria")
                evidence_list = [
                    {"text": ev["abstract"], "pmid": ev["pmid"]}
                    for ev in generation_result["supporting_evidence"]
                ]
                evidence_scores = score_by_bradford_hill_aging(evidence_list)

            # Stage 3: Identify aging hallmarks
            hallmarks_identified = None
            if identify_hallmarks:
                logger.info("Identifying aging hallmarks in evidence")
                hallmarks_identified = {}
                for i, evidence in enumerate(generation_result["supporting_evidence"]):
                    text = evidence["abstract"]
                    hallmarks = identify_aging_hallmarks(text)
                    if hallmarks:
                        hallmarks_identified[f"paper_{i + 1}"] = hallmarks

            # Compile results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results = {
                "query": query,
                "generated_hypotheses": generation_result["generated_hypotheses"],
                "supporting_evidence": generation_result["supporting_evidence"],
                "evidence_quality_scores": evidence_scores,
                "identified_hallmarks": hallmarks_identified,
                "metadata": {
                    "model": self.generation_model,
                    "top_k": top_k,
                    "pipeline_duration_seconds": duration,
                    "timestamp": datetime.now().isoformat(),
                    "num_documents_indexed": len(self.docs),
                },
            }

            logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            return results

        except Exception as e:
            raise LongevityException(f"Pipeline execution failed: {str(e)}", sys)

    def run_batch(
        self, queries: List[str], output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Run pipeline on multiple queries.

        Args:
            queries: List of research questions
            output_dir: Optional directory to save individual results

        Returns:
            List of result dictionaries
        """
        logger.info(f"Running batch pipeline for {len(queries)} queries")

        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            try:
                result = self.run(query)
                results.append(result)

                # Save individual result if output_dir specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    filename = f"result_{i:03d}_{query[:50].replace(' ', '_')}.json"
                    filepath = output_path / filename

                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    logger.info(f"Saved result to {filepath}")

            except Exception as e:
                logger.error(f"Failed to process query '{query}': {str(e)}")
                results.append({"query": query, "error": str(e)})

        logger.info(f"Batch pipeline completed. {len(results)} results.")
        return results


def save_results(results: Dict, output_path: str):
    """Save pipeline results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the complete longevity hypothesis generation pipeline"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Research question (e.g., 'Can NAD+ boosters extend healthspan?')",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of papers to retrieve"
    )
    parser.add_argument(
        "--model", default="google/flan-t5-base", help="Generation model name"
    )
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument(
        "--no-scoring", action="store_true", help="Skip evidence scoring"
    )
    parser.add_argument(
        "--no-hallmarks", action="store_true", help="Skip hallmark identification"
    )

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = LongevityInferencePipeline(generation_model=args.model, use_cuda=True)

    results = pipeline.run(
        query=args.query,
        top_k=args.top_k,
        score_evidence=not args.no_scoring,
        identify_hallmarks=not args.no_hallmarks,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("LONGEVITY HYPOTHESIS GENERATION PIPELINE RESULTS")
    print("=" * 80)
    print(f"\nQuery: {results['query']}")
    print(f"Duration: {results['metadata']['pipeline_duration_seconds']:.2f}s")

    print("\n" + "-" * 80)
    print("HYPOTHESES:")
    print("-" * 80)
    print(results["generated_hypotheses"])

    if results["evidence_quality_scores"]:
        print("\n" + "-" * 80)
        print("EVIDENCE QUALITY (Bradford Hill Criteria):")
        print("-" * 80)
        for criterion, score in results["evidence_quality_scores"].items():
            print(f"  {criterion:.<30} {score:.3f}")

    if results["identified_hallmarks"]:
        print("\n" + "-" * 80)
        print("AGING HALLMARKS IDENTIFIED:")
        print("-" * 80)
        for paper, hallmarks in results["identified_hallmarks"].items():
            print(f"\n  {paper}:")
            for h in hallmarks:
                print(f"    • {h['hallmark']}: {h['keyword']}")

    print("\n" + "=" * 80 + "\n")

    # Save if requested
    if args.output:
        save_results(results, args.output)
