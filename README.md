# Rand Line Optimizer — ML Model Overview

This repository predicts per-time-bin dining swipe probabilities for students and aggregates them into a demand surface the optimizer can use. Below is a guide to how the ML pipeline is built, trained, and used. This was initially built for a Palantir hackathon at Vanderbilt.

## What the model does
- Predicts the probability that a given student will swipe in a specific 15‑minute bin across the day.
- Incorporates historical behavior, schedule constraints (classes), and meal plan budgets.
- Produces a demand matrix that downstream services can aggregate for planning and scenario analysis.

## Key modules
- `feature_builder.py`: Builds the training matrix by joining:
  - Meal propensity priors per student (`MealPropensityService`)
  - Time-bin features from `TimeGridService` (meal periods, minutes from meal center, normalized bin index)
  - Schedule context via `AvailabilityService` (in-class boolean, free-time gaps before/after)
  - Meal plan features (weekly allowance, unlimited, flex)
- `meal_propensity_service.py`: Computes baseline propensities for breakfast, lunch, dinner per student from swipe history. Falls back to population averages when history is missing.
- `ml_model_trainer.py`: Trains a classifier (default Gradient Boosting; Logistic Regression optional) to classify whether probability > 0 per bin, then uses calibrated probabilities and post-processing to respect meal plan budgets.

## Training data construction
1. Build a time grid via `TimeGridService` (meal period windows and bin indices).
2. Compute student meal propensities from historical `swipes_data` in `MealPropensityService`.
3. Compute schedule availability from enrollments via `AvailabilityService` and `SectionsCatalog`.
4. `FeatureBuilder.build_training_matrix(student_ids)` creates one row per `(student_id, bin_index)` with features:
   - Time-bin: `meal_period_bin`, `minutes_from_meal_center`, `bin_index_normalized`
   - Meal propensity: `p_breakfast`, `p_lunch`, `p_dinner`
   - Plan: `weekly_allowance`, `unlimited`, `is_flex`
   - Schedule: `has_class_in_bin`, `free_gap_before_min`, `free_gap_after_min`, `is_large_gap`
   - Weak label: `target_probability` computed by distributing a student’s meal-period propensity into bins, weighted by free-time gaps and distance to meal center; bins in class get 0.

## Enhanced labeling and calibration
`MLModelTrainer.prepare_training_data` refines weak labels to create `enhanced_target`:
- Re-weights bins within each meal period by combining free-gap and distance-to-center weights and normalizes within the period, scaled by the student’s meal-period propensity.
- Enforces class constraints by zeroing bins when the student is in class.
- The final `target_probability` is the enhanced target.

## Model choices
- Default: `GradientBoostingClassifier` with tunable `n_estimators`, `max_depth`, `learning_rate`.
- Alternative: `LogisticRegression` with `C`, `max_iter`.
- Features are numerically encoded; categorical `meal_period_bin` is mapped to {Breakfast=1, Lunch=2, Dinner=3, None=0}. Booleans are cast to ints.

## Train/validate metrics
The trainer splits data into train/test (default 80/20 stratified by positive label) and reports:
- AUC on train/test
- LogLoss on train/test
- Feature importance (tree importances or absolute coefficients)

## Meal plan constraint calibration (inference)
Predicted probabilities are scaled by a daily budget factor derived from meal plans:
- Unlimited plans use a high cap; limited plans scale by `weekly_allowance / 7` normalized to ~3 meals/day.
- This ensures probabilities respect realistic per-day meal limits.

## Demand matrix generation
`predict_demand_matrix(student_ids)` iterates all bins and students to produce:
- Columns: `student_id`, `bin_index`, `swipe_probability`
This matrix can be aggregated to site/venue demand curves by time.

## Running the training demo
You can run the self-contained demos to build features and train the model using the sample CSVs in the repo.
The demo was AI generated in a few minutes just so I had something to show judges, so there is a bug or two in the display.

```bash
# Option 1: FeatureBuilder demo (prints feature summaries)
python feature_builder.py

# Option 2: ML trainer demo (prints metrics and sample predictions)
python ml_model_trainer.py
```

Both demos load data via `dataloader.py`, build the catalog and services, then construct features and train.

## Saving and loading models
`MLModelTrainer.save_model(filepath)` saves the calibrated model, feature columns, constraints, and config via `joblib`. Load with `load_model(filepath)` for inference.

## Inputs and data sources
- `students_*.csv`: Student roster(s)
- `class_enrollments.csv`: Enrollment and schedule data
- `swipes_data.csv`: Historical dining swipes
- The services reconcile these into features per `(student, bin)`.

## Design notes and assumptions
- Weak labels are heuristic but consistent and monotonic with intuitive drivers (availability, proximity to meal center).
- Calibration applies budget realism at inference rather than hard label balancing, keeping training simple and flexible.
- Population backfill avoids dropping students without history.

## Extending the model
- Add plan-specific features (e.g., rollover balance) via a `MealPlanNormalizer` integration.
- Add venue/location choice modeling (multinomial) on top of the per-bin swipe probability.
- Replace heuristic weak labels with logged implicit signals (e.g., app opens, queue sensor data) when available.
- Introduce temporal generalization (day-of-week, semester weeks) via additional features.

---
For a quick end-to-end example, see the `main()` functions in `feature_builder.py` and `ml_model_trainer.py`.
