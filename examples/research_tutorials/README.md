1. [Introduction to Research Module](./01_introduction.ipynb)
    * Research with one callable
    * Constructing domains
    * Generators in research
    * Instances of some class in Research
    * Parallel executions
    * Branches

2. [Advanced Usage of Domain](./02_domain.ipynb)
    * Basic usage
    * Operations
    * Samplers in Domain
    * Domains with Weights

3. [Research results processing](./03_results_processing.ipynb)
    * `to_df` parameters
    * Loading
    * Filtering

4. [Research with pipelines](./04_research_with_pipelines.ipynb)
    * Basic examples
        * 1 pipeline with fixed parameters
            * creating research
            * running several repetitions of an experiment
            * viewing research results
    * Running experiments with different parameters aka domain
        * 1 pipeline with variable parameters
            * creating and viewing domains
            * viewing filtered research results
    * More complex execution strategies
        * 2 pipelines, train & test + function + domain
            * adding test pipeline
            * defining pipeline execution frequency

5. [Advanced Research module usage](./05_advanced_usage_of_research.ipynb)
    * Reducing extra dataset loads
        * 1 pipeline with root and branch + domain
    * Performance
        * execution tasks managing
    * Cross-validation

6. [Domain update](./06_update_domain.ipynb)
