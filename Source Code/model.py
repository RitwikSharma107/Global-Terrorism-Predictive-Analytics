########################################################################################################
# Name: Ritwik Sharma (G01373222) and Mukund Sharma (G01374620)
########################################################################################################

# Import Libraries

import re
import string
import warnings
from functools import reduce

import pyspark.sql.functions as f
from pyspark.ml.clustering import LDA
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, StringIndexer, Tokenizer, HashingTF, CountVectorizer, \
    VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from xgboost.spark import SparkXGBClassifier

warnings.filterwarnings("ignore")

# Create a SparkSession
spark = SparkSession.builder.appName('final_project').getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

########################################################################################################
# Global Parameters

# Open a text file for writing
output_file = open('/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Output/output_model.txt', 'w')

# Instantiate the stopwords
stopwords = StopWordsRemover.loadDefaultStopWords("english")


########################################################################################################
# Functions

# Function: To clean text for the required column
def clean_text(df, feature):
    # Remove null values
    df = df.filter((df[feature].isNotNull()))
    # Remove punctuation and then convert to lower-case words
    df = df.withColumn(feature, f.lower(f.regexp_replace(feature, '[^a-zA-Z0-9]', ' ')))
    df = df.withColumn(feature, f.split(feature, ' '))
    # Convert to string
    df = df.withColumn(feature, f.concat_ws(' ', feature))
    # # Remove all numbers
    df = df.withColumn(feature, f.regexp_replace(feature, '\d+', ''))
    # Replace dots with spaces
    df = df.withColumn(feature, f.regexp_replace(feature, '\.', ' '))
    # Remove special characters
    df = df.withColumn(feature, f.regexp_replace(feature, '[%s]' % re.escape(string.punctuation), ''))
    # Get rid of double spaces
    df = df.withColumn(feature, f.regexp_replace(feature, ' +', ' '))
    return df


# Function: To encode categorical columns
def encode_cat_col(df, column):
    indexer = StringIndexer(inputCol=column, outputCol=f"{column}_encoded")
    df = indexer.fit(df).transform(df)
    return df.drop(column)


# Function: To encode categorical list columns
def encode_list_cat_cols(df, column):
    exploded_values = df.withColumn("value", f.explode(f.col(column)))
    distinct_values = exploded_values.select("value").distinct().rdd.flatMap(lambda x: x).collect()
    encode_value = lambda df, value: df.withColumn(value, f.array_contains(column, value).cast('int'))
    df = reduce(encode_value, distinct_values, df)
    return df.drop(column)


# Function: To implement predictive modeling using cross validation
def classification_analysis(df, feature_cols, label_col):
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = vector_assembler.transform(df)

    xgboost = SparkXGBClassifier(label_col=label_col, features_col="features")
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)

    # Define the parameter grid for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(xgboost.max_depth, [1, 2]) \
        .addGrid(xgboost.n_estimators, [10, 20]) \
        .addGrid(xgboost.learning_rate, [0.01, 0.1]) \
        .build()

    # Set up the CrossValidator
    crossval = CrossValidator(estimator=xgboost,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=10)

    # Split the data into training and testing
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Fit the CrossValidator to the training data
    cv_model = crossval.fit(train_data)

    # Make predictions on the test data using the best model found by cross-validation
    test_results = cv_model.transform(test_data)

    # Evaluate the model using the BinaryClassificationEvaluator
    roc_auc = evaluator.evaluate(test_results, {evaluator.metricName: "areaUnderROC"})
    return roc_auc, cv_model


########################################################################################################
### Main Funtion ###
def main():
    # Read CSV file
    df = spark.read.csv('/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Data/GTD.csv', header=True, inferSchema=True)

    ######################
    ### Topic Modeling ###
    ######################

    # Clean summary column
    df_summary = reduce(lambda df, feature: clean_text(df, feature), ['summary'], df)

    # Tokenize the categoriacal columns into words
    tokenizer = Tokenizer(inputCol='summary', outputCol=f"{'summary'}_words")
    df_summary = tokenizer.transform(df_summary)

    # Create a StopWordsRemover instance
    remover = StopWordsRemover(inputCol=f"{'summary'}_words", outputCol=f"filtered_{'summary'}_words",
                               stopWords=stopwords)
    df_summary = remover.transform(df_summary)

    # Hashing
    hashingTF = HashingTF(inputCol=f"filtered_{'summary'}_words", outputCol=f"{'summary'}_features", numFeatures=50)
    df_summary = hashingTF.transform(df_summary)

    # Vectorization - CountVectorizer
    countVectorizer = CountVectorizer(inputCol=f"filtered_{'summary'}_words", outputCol=f"{'summary'}_vectorized")
    cv_model = countVectorizer.fit(df_summary)
    df_summary = cv_model.transform(df_summary)

    # Remove the specified columns
    df_summary = df_summary.drop('summary', f"{'summary'}_words", f"filtered_{'summary'}_words",
                                 f"{'summary'}_features")

    ### LDA Model Tuning ###
    output_file.write('\nTopic Modeling\n')

    # Define the range of topics to evaluate
    num_topics_range = range(5, 21, 5)

    # Define a lambda function to train LDA models and calculate perplexity
    train_lda_model = lambda num_topics: LDA(k=num_topics, maxIter=10, featuresCol="summary_vectorized").fit(df_summary)

    # Train LDA models for different numbers of topics
    lda_models = list(map(train_lda_model, num_topics_range))

    # Calculate perplexity for each model
    perplexities = list(map(lambda model: model.logPerplexity(df_summary), lda_models))

    # Find the best model based on perplexity
    best_model_index = perplexities.index(min(perplexities))
    best_model = lda_models[best_model_index]
    best_num_topics = num_topics_range[best_model_index]
    lda_models_perplexities = list(
        map(lambda num, perp: f"Number of Topics: {num}, Perplexity: {perp}", num_topics_range, perplexities))

    output_file.write("\n".join(map(str, lda_models_perplexities)))
    output_file.write(f"\nBest Model - Number of Topics: {best_num_topics}")

    df_summary = best_model.transform(df_summary)

    ### Distribution of words most important for each topic ###
    output_file.write('\n\nDistribution of words most important for each topic:\n')
    topics = best_model.describeTopics(maxTermsPerTopic=10)
    topic_words = []
    topic_words = list(map(lambda i: [cv_model.vocabulary[idx] for idx in
                                      topics.filter(topics.topic == i).select("termIndices").first()[0]],
                           range(best_num_topics)))
    topic_top_words = list(
        map(lambda i, words: (f"Topic {i} - Top Words: {', '.join(words)}"), range(best_num_topics), topic_words))
    output_file.write("\n".join(map(str, topic_top_words)))

    ###########################
    ### Predictive Modeling ###
    ###########################

    output_file.write('\n\nPredictive Modeling\n')

    agg_cols = ['iyear', 'imonth', 'country', 'country_txt', 'region', 'region_txt', 'provstate', 'city',
                'attacktype1_txt', 'targtype1_txt', 'natlty1', 'natlty1_txt', 'gname', 'weaptype1_txt', 'success']

    df_model = df.withColumn("num_events", f.count("eventid").over(Window.partitionBy("gname"))) \
        .filter(
        (f.col('imonth') != 0) &
        (f.col('attacktype1') != 0) &
        (f.col('weaptype1') != -9) &
        (f.col('weaptype1') != 0) &
        (f.col('attacktype1_txt') != 'Unknown') &
        (f.col('targtype1_txt') != 'Unknown') &
        (f.col('weaptype1_txt') != 'Unknown') &
        (f.col('gname') != 'Unknown') &
        (f.col('num_events') >= 1000)) \
        .select(agg_cols) \
        .dropna() \
        .distinct()
    df_model = df_model.withColumn('weaptype1_txt', f.when(
        df['weaptype1_txt'] == 'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
        'Vehicle').otherwise(df['weaptype1_txt']))
    df_model = df_model.withColumn('success', f.col('success').cast("float"))

    df_model_grouped = df_model.groupby("region", "country", "country_txt", "provstate", "city", "iyear", "imonth",
                                        "success").agg(
        f.collect_set("attacktype1_txt").alias("attacktype1_txt"),
        f.count("attacktype1_txt").alias("num_of_attacks"),
        f.collect_set("targtype1_txt").alias("targtype1_txt"),
        f.count("targtype1_txt").alias("num_of_targets"),
        f.collect_set("natlty1_txt").alias("natlty1_txt"),
        f.count("natlty1_txt").alias("num_of_target_natltys"),
        f.collect_set("gname").alias("gname"),
        f.count("gname").alias("num_of_gnames"),
        f.collect_set("weaptype1_txt").alias("weaptype1_txt"),
        f.count("weaptype1_txt").alias("num_of_weapons"),
    ).orderBy(
        f.asc("region"),
        f.asc("country"),
        f.asc("provstate"),
        f.asc("city"),
        f.asc("iyear"),
        f.asc("imonth")
    )

    # Encode Categorical Columns
    df_model_grouped = encode_cat_col(df_model_grouped, 'provstate')
    df_model_grouped = encode_cat_col(df_model_grouped, 'city')
    df_model_grouped = encode_list_cat_cols(df_model_grouped, 'attacktype1_txt')
    df_model_grouped = encode_list_cat_cols(df_model_grouped, 'targtype1_txt')
    df_model_grouped = encode_list_cat_cols(df_model_grouped, 'natlty1_txt')
    df_model_grouped = encode_list_cat_cols(df_model_grouped, 'weaptype1_txt')
    df_model_grouped = encode_list_cat_cols(df_model_grouped, 'gname')

    # Model Perfomance Evaluation
    output_file.write('\nXGBoostClassifier')
    label_col = 'success'
    feature_cols = list(filter(lambda col: col not in ['success', 'country_txt'], df_model_grouped.columns))
    roc_auc, cv_model = classification_analysis(df_model_grouped, feature_cols, label_col)
    output_file.write(f"\nROC AUC: {round(roc_auc, 4)}\n")

    ################################################
    ### Predictive Modeling with External Factor ###
    ################################################

    # Load Political Stabilty Data
    df_pol_stable = spark.read.csv('/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Data/Political_Stabilty.csv',
                                   header=True, inferSchema=True)

    # Preprocess Political Stabilty Data
    df_pol_stable = df_pol_stable.filter(
        f.col('Series Name') == 'Political Stability and Absence of Violence/Terrorism: Estimate')
    cols_to_keep = ["Country Name", "Country Code", "Series Name", "Series Code"]
    year_cols = [c for c in df_pol_stable.columns if c not in cols_to_keep]
    df_pol_stable = df_pol_stable.withColumn("years", f.array(*year_cols))
    df_pol_stable = df_pol_stable.select("Country Name", "Country Code", "Series Name", "Series Code",
                                         f.explode(f.col("years")).alias("Political_Stabilty"))
    df_pol_stable = df_pol_stable.groupBy(f.col('Country Name').alias('country_match')) \
        .agg(f.round(f.avg('Political_Stabilty'), 4).alias('Political_Stabilty')) \
        .sort(f.col('Country Name'))

    # Join dataframes to get politcial stability for each country
    df_model_grouped = df_model_grouped.join(f.broadcast(df_pol_stable),
                                             df_model_grouped['country_txt'] == df_pol_stable['country_match'], 'left')
    df_model_grouped = df_model_grouped.withColumn('Political_Stabilty',
                                                   f.when(f.col('Political_Stabilty').isNull(), 0).otherwise(
                                                       f.col('Political_Stabilty')))

    # Model Perfomance Evaluation (With Political Stabilty)
    output_file.write('\nXGBoostClassifier (With Political Stabilty)')
    label_col = 'success'
    feature_cols = list(
        filter(lambda col: col not in ['success', 'country_txt', 'country_match'], df_model_grouped.columns))
    roc_auc, cv_model = classification_analysis(df_model_grouped, feature_cols, label_col)
    output_file.write(f"\nROC AUC: {round(roc_auc, 4)}\n")


if __name__ == "__main__":
    main()

# Close the output file
output_file.close()

# Stop Spark session
spark.stop()

########################################################################################################
### End ###
########################################################################################################