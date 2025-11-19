"""
Training pipeline for optimizing validator weights.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from app.validators import HybridValidator
from app.storage import create_store


class TrainingPipeline:
    """Train validator scoring weights based on labeled datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize training pipeline.
        
        Args:
            config: Training configuration
        """
        if config is None:
            from app.config import get_config
            config = get_config().training
        
        self.config = config
        self.validator = HybridValidator()
        self.store = create_store("json", {"path": "data/training_results.json"})
        
    def load_dataset(self, dataset_name: str, split: str = "train", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load a validation dataset.
        
        Args:
            dataset_name: Name of dataset to load
            split: Dataset split
            max_samples: Maximum number of samples to load
            
        Returns:
            List of dataset samples
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            if dataset_name.lower() == "halueval":
                # Load HaluEval dataset
                dataset = load_dataset("pminervini/HaluEval", split=split)
            elif dataset_name.lower() == "summeval":
                # Load SummEval dataset
                dataset = load_dataset("mteb/summeval", split="test")
            else:
                logger.warning(f"Unknown dataset: {dataset_name}, using mock data")
                return self._create_mock_dataset(max_samples or 100)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            return [dict(item) for item in dataset]
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            logger.info("Using mock dataset for demonstration")
            return self._create_mock_dataset(max_samples or 100)
    
    def _create_mock_dataset(self, size: int) -> List[Dict[str, Any]]:
        """Create a mock dataset for testing."""
        return [
            {
                "prompt": f"Sample prompt {i}",
                "text": f"Sample response {i} with varying quality.",
                "label": np.random.choice([0, 1], p=[0.3, 0.7])  # 70% good
            }
            for i in range(size)
        ]
    
    def evaluate_on_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate validator on dataset.
        
        Args:
            dataset: List of validation samples
            
        Returns:
            Evaluation metrics
        """
        predictions = []
        labels = []
        scores = []
        
        for sample in tqdm(dataset, desc="Evaluating"):
            prompt = sample.get("prompt", "")
            text = sample.get("text", sample.get("response", ""))
            label = sample.get("label", 1)
            
            # Validate
            result = self.validator.validate(text, prompt=prompt)
            
            predictions.append(1 if result.passed else 0)
            labels.append(label)
            scores.append(result.score)
        
        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)
        scores = np.array(scores)
        
        accuracy = np.mean(predictions == labels)
        precision = np.sum((predictions == 1) & (labels == 1)) / max(np.sum(predictions == 1), 1)
        recall = np.sum((predictions == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores))
        }
    
    def optimize_weights(
        self,
        train_dataset: List[Dict[str, Any]],
        val_dataset: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize validator weights using gradient-free optimization.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of optimization epochs
            
        Returns:
            Optimization results
        """
        logger.info("Starting weight optimization")
        
        best_weights = self.validator.weights.copy()
        best_score = 0.0
        history = []
        
        for epoch in range(epochs):
            # Try random perturbations
            perturbation = {
                "heuristics": np.random.normal(0, 0.05),
                "judge": np.random.normal(0, 0.05)
            }
            
            # Apply perturbation
            new_weights = {
                k: max(0.0, min(1.0, v + perturbation[k]))
                for k, v in best_weights.items()
            }
            
            # Normalize weights
            total = sum(new_weights.values())
            new_weights = {k: v / total for k, v in new_weights.items()}
            
            # Test new weights
            self.validator.weights = new_weights
            metrics = self.evaluate_on_dataset(train_dataset[:100])  # Sample for speed
            
            score = metrics["f1"]
            
            # Track progress
            epoch_result = {
                "epoch": epoch,
                "weights": new_weights.copy(),
                "metrics": metrics,
                "score": score
            }
            history.append(epoch_result)
            
            # Update best
            if score > best_score:
                best_score = score
                best_weights = new_weights.copy()
                logger.info(f"Epoch {epoch}: New best F1 = {score:.4f}")
        
        # Set best weights
        self.validator.weights = best_weights
        
        # Final evaluation
        final_train_metrics = self.evaluate_on_dataset(train_dataset)
        final_val_metrics = self.evaluate_on_dataset(val_dataset) if val_dataset else {}
        
        result = {
            "best_weights": best_weights,
            "best_score": best_score,
            "train_metrics": final_train_metrics,
            "val_metrics": final_val_metrics,
            "history": history
        }
        
        # Save results
        self.store.save({
            "type": "training_run",
            "config": self.config,
            "result": result
        })
        
        logger.info(f"Training complete. Best F1: {best_score:.4f}")
        logger.info(f"Best weights: {best_weights}")
        
        return result
    
    def run_training(self) -> Dict[str, Any]:
        """Run complete training pipeline."""
        datasets_config = self.config.get("datasets", [])
        
        all_train_data = []
        all_val_data = []
        
        # Load all configured datasets
        for ds_config in datasets_config:
            if not ds_config.get("enabled", True):
                continue
            
            name = ds_config["name"]
            split = ds_config.get("split", "train")
            max_samples = ds_config.get("max_samples")
            
            data = self.load_dataset(name, split, max_samples)
            
            # Split into train/val
            val_split = self.config.get("validation_split", 0.2)
            split_idx = int(len(data) * (1 - val_split))
            
            all_train_data.extend(data[:split_idx])
            all_val_data.extend(data[split_idx:])
        
        logger.info(f"Total training samples: {len(all_train_data)}")
        logger.info(f"Total validation samples: {len(all_val_data)}")
        
        # Run optimization
        return self.optimize_weights(
            all_train_data,
            all_val_data,
            epochs=self.config.get("epochs", 10)
        )
