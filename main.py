import mlflow
import time
from config import logger
from scraping import scrape_reviews
from preprocessing import TextPreprocessor
from modeling import generate_lda_model
from utils import load_abbreviation_dict

class ReviewLensApp:
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        
        # Enhanced MLflow initialization with error handling
        try:
            # Set the MLflow tracking URI
            tracking_uri = "http://127.0.0.1:5000"
            mlflow.set_tracking_uri(tracking_uri)
            
            # Verify MLflow connection
            try:
                mlflow.search_experiments(max_results=1)
                logger.info("Successfully connected to MLflow tracking server")
            except Exception as e:
                logger.error(f"Failed to connect to MLflow tracking server at {tracking_uri}")
                raise ConnectionError(f"MLflow connection failed: {str(e)}")

            # Ensure the experiment exists
            experiment_name = "ReviewLensApp"
            logger.info("Starting MLflow experiment setup...")  # Added log for visibility
            existing_experiment = mlflow.get_experiment_by_name(experiment_name)
            if existing_experiment is None:
                mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                logger.info(f"Experiment already exists: {experiment_name}")  # Added log for existing experiment
                
            mlflow.set_experiment(experiment_name)
            
            # Log tracking information
            tracking_uri = mlflow.get_tracking_uri()
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"MLflow Tracking URI: {tracking_uri}")
            logger.info(f"Experiment ID: {experiment_id}")
            
        except Exception as e:
            logger.critical(f"MLflow initialization failed: {str(e)}")
            raise

    def process_app_url(self, url):
        """Process a Google Play Store URL and return analysis results"""
        try:
            with mlflow.start_run():  # Start an MLflow run
                start_time = time.time()  # Start timing the entire process

                # Step 1: Scrape reviews
                start_time_scrape = time.time()
                df = scrape_reviews(url)
                scrape_duration = time.time() - start_time_scrape
                mlflow.log_metric("scrape_duration_seconds", scrape_duration)
                mlflow.log_metric("scrape_requests_total", 1)
                mlflow.log_metric("reviews_processed_total", len(df))
                logger.info(f"Successfully scraped {len(df)} reviews in {scrape_duration:.2f} seconds")
                
                # Step 2: Load abbreviation dictionary
                abbreviation_dict = load_abbreviation_dict()
                
                # Step 3: Preprocess reviews
                start_time_preprocess = time.time()
                processed_df = self.text_preprocessor.preprocess_reviews(df, abbreviation_dict)
                preprocess_duration = time.time() - start_time_preprocess
                mlflow.log_metric("preprocess_duration_seconds", preprocess_duration)
                mlflow.log_artifact("cleaned_review.csv")  # Log cleaned dataset as artifact
                logger.info(f"Successfully preprocessed {len(processed_df)} reviews in {preprocess_duration:.2f} seconds")
                
                # Step 4: Generate LDA model
                start_time_lda = time.time()
                model, dictionary, corpus = generate_lda_model(processed_df)
                lda_duration = time.time() - start_time_lda
                mlflow.log_metric("lda_duration_seconds", lda_duration)
                logger.info(f"LDA model generation complete in {lda_duration:.2f} seconds")
                
                # Log total duration
                total_duration = time.time() - start_time
                mlflow.log_metric("total_process_duration_seconds", total_duration)
                logger.info(f"Total process duration: {total_duration:.2f} seconds")
                
                # Log additional metadata for better tracking
                mlflow.set_tag("source_url", url)
                mlflow.log_param("model_type", "LDA")
                
                return {
                    "model": model,
                    "dictionary": dictionary,
                    "corpus": corpus,
                    "processed_data": processed_df
                }
        except Exception as e:
            mlflow.log_metric("scrape_errors_total", 1)
            logger.error(f"Error processing URL: {str(e)}")
            raise
    
    def run_cli(self):
        """Run as a command-line application"""
        while True:
            try:
                url = input("Enter Google Play Store URL (or 'quit' to exit): ")
                if url.lower() == 'quit':
                    break

                result = self.process_app_url(url)
                
                # Print a summary of results
                if result and result["model"]:
                    print("\nAnalysis completed successfully!")
                    print(f"Number of reviews analyzed: {len(result['processed_data'])}")
                    
                    # Print top words for each topic
                    model = result["model"]
                    for i in range(model.num_topics):
                        topic_words = model.show_topic(i, topn=10)
                        words = [word for word, _ in topic_words]
                        print(f"Topic {i+1}: {', '.join(words)}")
                    
                    print("\nWord clouds saved in the 'static/wordclouds' directory")
                    
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"An error occurred: {str(e)}. Please check the logs for details.")

# Keep the old method for backward compatibility
def run(self):
    """Alias for run_cli for backward compatibility"""
    return self.run_cli()

# Add the run method to the class
ReviewLensApp.run = run

if __name__ == "__main__":
    app = ReviewLensApp()
    app.run_cli()