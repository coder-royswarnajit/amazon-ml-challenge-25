# Implementation Plan

- [x] 1. Project setup and configuration





  - Create project directory structure (data/, models/, checkpoints/, logs/, src/, scripts/, tests/)
  - Create requirements.txt with all dependencies
  - Create config.py with centralized configuration (paths, hyperparameters, model settings)
  - Create setup.py for package installation
  - Initialize git repository with .gitignore
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 1.1 Write property test for configuration completeness


  - **Property 38: Configuration path completeness**
  - **Property 39: Configuration hyperparameter completeness**
  - **Property 40: Feature engineering config completeness**
  - **Validates: Requirements 10.2, 10.3, 10.4, 10.5**

- [x] 2. Implement checkpoint system





  - Create CheckpointManager class in src/utils/checkpoint.py
  - Implement save_checkpoint() with automatic cleanup
  - Implement load_checkpoint() with error handling
  - Implement get_latest_checkpoint() for resume logic
  - Add checkpoint type support (quick, full, minimal)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 2.1 Write property test for checkpoint round-trip


  - **Property 11: Checkpoint completeness**
  - **Property 12: Training resumption continuity**
  - **Validates: Requirements 3.2, 3.3**

- [x] 2.2 Write property test for checkpoint cleanup


  - **Property 13: Checkpoint cleanup policy**
  - **Validates: Requirements 3.4**

- [x] 3. Implement data download module





  - Create ResumableImageDownloader class in src/data/downloader.py
  - Implement download_batch() with parallel workers
  - Implement _download_single() with retry logic and exponential backoff
  - Implement _save_progress() and _load_progress() for resume capability
  - Add image verification functionality
  - Add automatic dataset download from competition source
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3.1 Write property test for batch download consistency


  - **Property 1: Batch download consistency**
  - **Validates: Requirements 1.1**

- [x] 3.2 Write property test for resume from interruption

  - **Property 2: Resume from interruption**
  - **Validates: Requirements 1.2**

- [x] 3.3 Write property test for retry with exponential backoff

  - **Property 3: Retry with exponential backoff**
  - **Validates: Requirements 1.3**

- [x] 3.4 Write property test for corrupted image detection

  - **Property 4: Corrupted image detection**
  - **Validates: Requirements 1.4**

- [x] 4. Implement feature engineering module





  - Create FeatureEngineer class in src/data/feature_engineering.py
  - Implement extract_ipq_features() with regex patterns and unit normalization
  - Implement extract_text_statistics() for text metrics
  - Implement extract_keyword_features() for quality/discount keywords
  - Implement extract_brand_features() for brand extraction
  - Implement fit_tfidf() and transform_tfidf() for TF-IDF vectorization
  - Implement engineer_features() to orchestrate all feature extraction
  - Add save_features() and load_features() for serialization
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [x] 4.1 Write property test for IPQ extraction determinism


  - **Property 5: IPQ extraction determinism**
  - **Validates: Requirements 2.1**

- [x] 4.2 Write property test for text statistics correctness


  - **Property 6: Text statistics correctness**
  - **Validates: Requirements 2.2**

- [x] 4.3 Write property test for keyword detection


  - **Property 7: Keyword detection completeness**
  - **Validates: Requirements 2.3**

- [x] 4.4 Write property test for TF-IDF consistency


  - **Property 8: TF-IDF transformation consistency**
  - **Validates: Requirements 2.6**

- [x] 4.5 Write property test for unit normalization


  - **Property 9: Unit normalization reversibility**
  - **Validates: Requirements 2.7**

- [x] 4.6 Write property test for feature serialization


  - **Property 10: Feature serialization round-trip**
  - **Validates: Requirements 2.8**

- [x] 5. Implement metrics and evaluation module





  - Create metrics module in src/utils/metrics.py
  - Implement calculate_smape() with proper formula
  - Implement smape_scorer() for sklearn compatibility
  - Implement calculate_metrics_by_quantile() for stratified analysis
  - Implement evaluate_predictions() for comprehensive evaluation
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 5.1 Write property test for SMAPE formula correctness


  - **Property 34: SMAPE formula correctness**
  - **Validates: Requirements 9.1**

- [x] 5.2 Write property test for log space conversion

  - **Property 35: Log space conversion**
  - **Validates: Requirements 9.2**

- [x] 5.3 Write property test for metrics completeness

  - **Property 36: Metrics completeness**
  - **Validates: Requirements 9.3**

- [x] 5.4 Write property test for quantile stratification

  - **Property 37: Quantile SMAPE stratification**
  - **Validates: Requirements 9.4**

- [x] 6. Implement PyTorch dataset and dataloader

  - Create AmazonMLDataset class in src/data/dataset.py
  - Implement __init__() to merge raw data with features
  - Implement __getitem__() with text tokenization, image loading, and tabular processing
  - Add image transformation with augmentation for training mode
  - Add NaN filling for tabular features
  - Add fallback for missing/corrupted images
  - Implement get_dataloader() with optimized settings
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [x] 6.1 Write property test for data merge correctness

  - **Property 28: Data merge correctness**
  - **Validates: Requirements 8.1**

- [x] 6.2 Write property test for tokenization length


  - **Property 29: Tokenization length constraint**
  - **Validates: Requirements 8.2**

- [x] 6.3 Write property test for image transformation
  - **Property 30: Image transformation shape**
  - **Validates: Requirements 8.3**

- [x] 6.4 Write property test for data augmentation
  - **Property 31: Data augmentation conditional application**
  - **Validates: Requirements 8.4**

- [x] 6.5 Write property test for NaN filling
  - **Property 32: NaN filling completeness**
  - **Validates: Requirements 8.5**

- [x] 6.6 Write property test for image loading fallback
  - **Property 33: Image loading fallback**
  - **Validates: Requirements 8.7**

- [x] 7. Implement custom loss functions
  - Create losses module in src/models/losses.py
  - Implement HuberSMAPELoss for neural network training
  - Implement FocalSMAPELoss as alternative loss
  - Implement lgb_smape_objective() and lgb_smape_eval() for LightGBM
  - Implement xgb_smape_objective() for XGBoost
  - _Requirements: 5.1, 6.1, 6.2_

- [x] 8. Implement multimodal model architecture
  - Create OptimizedMultimodalModel in src/models/multimodal.py
  - Implement text encoder (DeBERTa-small) with gradient checkpointing
  - Implement image encoder (EfficientNet-B2) with frozen early layers
  - Implement CrossModalAttention for bidirectional fusion
  - Implement GatedFusion as alternative fusion mechanism
  - Implement tabular projection layer
  - Implement regression head (3-layer MLP)
  - Add count_parameters() and count_trainable_parameters() methods
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 8.1 Write property test for forward pass shape consistency
  - **Property 15: Multimodal forward pass shape consistency**
  - **Validates: Requirements 4.5**

- [x] 8.2 Write property test for tabular projection
  - **Property 16: Tabular feature projection**
  - **Validates: Requirements 4.4**

- [x] 9. Implement model utilities
  - Create ModelEMA class in src/models/utils.py
  - Implement EMA update logic with decay factor
  - Implement apply_shadow() and restore() for evaluation
  - Add save_model() and load_model() helper functions
  - _Requirements: 5.6_

- [x] 9.1 Write property test for EMA update consistency
  - **Property 20: EMA update consistency**
  - **Validates: Requirements 5.6**

- [x] 10. Implement neural network training module
  - Create train_neural_network() in src/training/train_neural_net.py
  - Implement setup_lora() to apply LoRA fine-tuning
  - Implement training loop with mixed precision (FP16)
  - Add gradient accumulation (4 steps)
  - Add gradient clipping (max norm 1.0)
  - Add learning rate scheduling (linear warmup + decay)
  - Add EMA updates after each step
  - Add validation after each epoch
  - Add best model saving based on validation SMAPE
  - Add automatic checkpoint saving every 30 minutes
  - Implement predict() for generating predictions
  - Implement predict_with_tta() for test-time augmentation
  - _Requirements: 4.8, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

- [x] 10.1 Write property test for LoRA parameter efficiency
  - **Property 14: LoRA parameter efficiency**
  - **Validates: Requirements 4.8**
  - _Note: Test skipped if PEFT has compatibility issues_

- [x] 10.2 Write property test for learning rate schedule
  - **Property 17: Learning rate schedule monotonicity**
  - **Validates: Requirements 5.3**

- [x] 10.3 Write property test for gradient accumulation
  - **Property 18: Gradient accumulation correctness**
  - **Validates: Requirements 5.4**

- [x] 10.4 Write property test for gradient clipping
  - **Property 19: Gradient clipping enforcement**
  - **Validates: Requirements 5.5**

- [x] 10.5 Write property test for best model selection
  - **Property 21: Best model selection**
  - **Validates: Requirements 5.8**

- [x] 11. Implement GBDT training module
  - Create train_gbdt_models() in src/training/train_gbdt.py
  - Implement optimize_lightgbm() with Optuna
  - Implement train_lightgbm() with custom SMAPE objective
  - Implement optimize_xgboost() with Optuna
  - Implement train_xgboost() with custom SMAPE objective
  - Implement train_catboost() with built-in regression
  - Add model saving for all GBDT models
  - Add prediction generation for train/val/test splits
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 11.1 Write property test for GBDT model serialization
  - **Property 22: GBDT model serialization**
  - **Validates: Requirements 6.5**

- [x] 11.2 Write property test for GBDT prediction completeness
  - **Property 23: GBDT prediction completeness**
  - **Validates: Requirements 6.6**

- [x] 12. Implement ensemble training module
  - Create train_ensemble() in src/training/train_ensemble.py
  - Implement level-1 meta-feature stacking
  - Implement Ridge meta-learner training
  - Implement ElasticNet meta-learner training
  - Implement shallow LightGBM meta-learner training
  - Implement level-2 weight optimization using scipy.optimize
  - Implement isotonic regression calibration
  - Add ensemble artifact saving (meta-learners, weights, calibration)
  - Add performance reporting (individual models + ensemble)
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 12.1 Write property test for meta-feature stacking
  - **Property 24: Meta-feature stacking correctness**
  - **Validates: Requirements 7.1**

- [x] 12.2 Write property test for weight normalization
  - **Property 25: Level-2 weight normalization**
  - **Validates: Requirements 7.3**

- [x] 12.3 Write property test for isotonic calibration
  - **Property 26: Isotonic calibration monotonicity**
  - **Validates: Requirements 7.5**

- [x] 12.4 Write property test for ensemble artifact completeness
  - **Property 27: Ensemble artifact completeness**
  - **Validates: Requirements 7.6**

- [x] 13. Implement visualization module
  - Create visualization utilities in src/utils/visualization.py
  - Implement plot_training_curves() for loss and SMAPE plots
  - Implement plot_predictions() for scatter and residual plots
  - Implement plot_error_distribution() for error histograms
  - Add high-resolution PNG saving for all plots
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [x] 14. Create execution scripts for pipeline stages
  - Create scripts/run_stage1_setup.py for data download
  - Create scripts/run_stage2_features.py for feature engineering
  - Create scripts/run_stage3_neural_net.py for neural network training
  - Create scripts/run_stage4_gbdt.py for GBDT training
  - Create scripts/run_stage5_ensemble.py for ensemble training
  - Create scripts/create_submission.py for final submission generation
  - Add command-line argument parsing for all scripts
  - Add progress logging and error handling
  - _Requirements: 1.1-1.5, 2.1-2.8, 4.1-4.8, 5.1-5.9, 6.1-6.6, 7.1-7.7_

- [x] 15. Implement submission generation
  - Create create_submission.py script
  - Load best ensemble predictions for test set
  - Convert predictions from log space to original space
  - Create DataFrame with sample_id and predicted_price columns
  - Validate all test samples have predictions
  - Save submission CSV
  - Log output path and basic statistics
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 15.1 Write property test for submission format
  - **Property 41: Submission format correctness**
  - **Validates: Requirements 12.3**

- [x] 15.2 Write property test for submission completeness
  - **Property 42: Submission completeness**
  - **Validates: Requirements 12.4**

- [x] 15.3 Write property test for log space conversion
  - **Property 43: Submission log space conversion**
  - **Validates: Requirements 12.2**

- [x] 16. Create README and documentation
  - Write comprehensive README.md with project overview
  - Document installation instructions
  - Document usage instructions for each pipeline stage
  - Add hardware requirements and setup guide
  - Add troubleshooting section
  - Create API documentation for key modules
  - Add example outputs and expected results

- [x] 17. Checkpoint - Ensure all tests pass
  - All 94 tests collected, 91 passed, 3 skipped (GPU and PEFT tests)

- [x] 18. Create integration tests
  - Write end-to-end pipeline test with small dataset
  - Write checkpoint resume integration test
  - Write multi-stage pipeline test
  - Add test fixtures and utilities
  - _File: tests/test_integration.py_

- [x] 19. Create performance tests
  - Write memory usage test for 6GB GPU constraint
  - Write checkpoint size test
  - Write download speed test
  - Write feature engineering speed test
  - _File: tests/test_performance.py_

- [x] 20. Final validation and submission
  - Run complete pipeline on full dataset
  - Verify final validation SMAPE < 9%
  - Generate final submission file
  - Validate submission format
  - Create final results visualization
  - Document final metrics and model performance
  - _File: scripts/run_validation.py - ALL 8 VALIDATIONS PASSED_
