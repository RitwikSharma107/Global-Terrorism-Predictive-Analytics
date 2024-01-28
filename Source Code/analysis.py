########################################################################################################
# Name: Ritwik Sharma (G01373222) and Mukund Sharma (G01374620)
########################################################################################################

# Import Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import re
import string
import networkx as nx
from functools import reduce
from wordcloud import WordCloud

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType, IntegerType, FloatType

# Remove unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Create a SparkSession
spark = SparkSession.builder.appName('final_project_visualizations').getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

########################################################################################################
# Global Parameters

# Get matplotlib graphs with dark background
plt.style.use('dark_background')

# Open a text file for writing
output_file_analysis = open('/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Output/output_analysis.txt', 'w')
########################################################################################################
# Functions

# Function: To plot anomaly detection by column category
def plot_anomaly(df, column):
    mean_df = df.groupBy(column).agg(f.mean("nkill").alias("mean_kills"))
    std_dev = df.groupBy(column).agg(f.stddev("nkill").alias("std_dev_kills"))
    df = df.join(mean_df, column).join(std_dev, column)

    diff_col = f.abs(f.col("nkill") - f.col("mean_kills"))
    outliers_df = df.where(diff_col > 3 * f.col("std_dev_kills"))

    col_count = outliers_df.groupBy(column).count().sort("count", ascending=False).limit(10)
    col_count = col_count.selectExpr(column, "count").toPandas()

    bar_plot = px.bar(col_count, x=column, y="count", color=column,
                      color_discrete_sequence=px.colors.sequential.RdBu)
    bar_plot.update_layout(title_text=f"Top 10 Outlier {column} by Fatality Count", template='plotly_dark')
    bar_plot.write_html(f"/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_Outlier_{column}_by_Fatality_Count.html")


# Function: To clean text for specified column
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


# Function: To display Wordcloud
def display_wordcloud(df, column):
    text_list = df.select(column).rdd.flatMap(lambda x: x).collect()
    text = ' '.join(text_list)

    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='cividis').generate(text)
    plt.figure(figsize=(20, 12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud from {column} Column')
    plt.savefig(f'/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/{column}_Word_Cloud.png')


########################################################################################################
### Main Funtion ###
def main():
    # Read CSV file
    df = spark.read.csv('/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/drive/MyDrive/MS/Big Data/Final Project/GTD.csv', header=True, inferSchema=True)

    #################################
    ### Exploratory Data Analysis ###
    #################################

    # Shape of Data
    output_file_analysis.write(f'\nThere are {df.count()} records and {len(df.columns)} attributes')

    # Numerical Features
    numerical_features = list(map(lambda field: field.name,
                                  filter(lambda field: isinstance(field.dataType, (DoubleType, IntegerType, FloatType)),
                                         df.schema.fields)))
    output_file_analysis.write(f'\nNumber of numerical variables: {len(numerical_features)}')

    # Categorical Features
    categorical_features = list(map(lambda field: field.name,
                                    filter(lambda field: isinstance(field.dataType, (StringType)),
                                           df.schema.fields)))
    output_file_analysis.write(f'\nNumber of categorical variables: {len(categorical_features)}')

    # Percenatge of missing values for each feature
    output_file_analysis.write("\n\nPercenatge of missing values for each feature:\n")
    feature_missing_values = list(map(
        lambda col_name: (
            col_name,
            df.filter(f.col(col_name).isNull()).count(),
            df.filter(f.col(col_name).isNull()).count() / df.count()
        ),
        df.columns
    ))
    print_feature_missing = lambda \
        x: f"The feature '{x[0]}' has {x[1]} missing values with {round(x[2] * 100, 2)}% missing"
    output_file_analysis.write('\n'.join(list(map(print_feature_missing, feature_missing_values))))

    # Heatmap
    df_pandas = df.toPandas()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_pandas.corr(), cmap='rocket_r', annot=True, ax=ax)
    plt.title('Heatmap of correlation matrix', fontsize=20)
    plt.savefig('/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/heatmap_correlation.png')

    ### Terrorism Fatalities ###

    # Total Killings over the years
    show = df.groupBy('iyear') \
        .agg(f.sum('nkill').alias('total_nkill')) \
        .orderBy(f.desc('iyear')) \
        .collect()

    x_values = list(map(lambda row: row['iyear'], show))
    y_values = list(map(lambda row: row['total_nkill'], show))

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Total Killings over the years'])
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values, mode='lines+markers', name='Killings', line=dict(color='red')),
        row=1, col=1
    )
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Killings', template='plotly_dark')
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Total_Killings_over_the_years.html")

    # Total Killings over the years by Region
    show = df.groupBy('iyear', 'region_txt') \
        .agg(f.sum('nkill').alias('Killed')) \
        .orderBy('iyear', 'region_txt') \
        .collect()
    x_values = list(map(lambda row: row['iyear'], show))
    y_values = list(map(lambda row: row['Killed'], show))
    region_values = list(map(lambda row: row['region_txt'], show))

    fig = go.Figure()
    region_x = list(map(lambda region: list(
        map(lambda i: x_values[i[0]], filter(lambda i: region_values[i[0]] == region, enumerate(x_values)))),
                        set(region_values)))
    region_y = list(map(lambda region: list(
        map(lambda i: y_values[i[0]], filter(lambda i: region_values[i[0]] == region, enumerate(y_values)))),
                        set(region_values)))
    list(map(lambda region, x, y: fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=region)),
             set(region_values), region_x, region_y))
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Total Killings',
        title='Total Killings by Region Over Years',
        template='plotly_dark'
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Total_Killings_by_Region_Over_Years.html")

    # Top 10 countries affected by terrorism
    show = df.groupBy('country_txt') \
        .agg(f.sum('nkill').alias('total_nkill')) \
        .orderBy(f.desc('total_nkill')) \
        .limit(10) \
        .collect()
    x_values = list(map(lambda row: row['country_txt'], show))
    y_values = list(map(lambda row: row['total_nkill'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='red'
    ))
    fig.update_layout(
        xaxis_title='Country',
        yaxis_title='Total Killings',
        title='Top 10 countries affected by terrorism',
        template='plotly_dark'
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_countries_affected_by_terrorism.html")

    # Top 10 cities affected by terrorism
    show = df.filter(f.col('city') != 'Unknown') \
        .groupBy('city') \
        .agg(f.sum('nkill').alias('total_nkill')) \
        .orderBy(f.desc('total_nkill')) \
        .limit(10) \
        .collect()
    x_values = list(map(lambda row: row['city'], show))
    y_values = list(map(lambda row: row['total_nkill'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='red'
    ))
    fig.update_layout(
        xaxis_title='City',
        yaxis_title='Total Killings',
        title='Top 10 cities affected by terrorism',
        template='plotly_dark'
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_cities_affected_by_terrorism.html")

    # Total Killings by Terrorist Organizations over the years (Top 10)
    show = df.filter(df['gname'] != "Unknown").groupBy('iyear', 'gname') \
        .agg(f.sum('nkill').alias('Killed')) \
        .orderBy('iyear', f.desc('Killed')) \
        .collect()
    top_10 = list(map(lambda row: row['gname'], show[:10]))
    x_values = list(map(lambda row: row['iyear'], filter(lambda row: row['gname'] in top_10, show)))
    y_values = list(map(lambda row: row['Killed'], filter(lambda row: row['gname'] in top_10, show)))
    region_values = list(map(lambda row: row['gname'], filter(lambda row: row['gname'] in top_10, show)))

    fig = go.Figure()
    region_x = list(map(lambda region: list(
        map(lambda i: x_values[i], filter(lambda i: region_values[i] == region, range(len(x_values))))),
                        set(region_values)))
    region_y = list(map(lambda region: list(
        map(lambda i: y_values[i], filter(lambda i: region_values[i] == region, range(len(y_values))))),
                        set(region_values)))
    list(map(lambda region, x, y: fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=region)),
             set(region_values), region_x, region_y))
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Total Killings',
        title='Total Killings by Terrorist Organizations over the years (Top 10)',
        template='plotly_dark'
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Total_Killings_by_Terrorist_Organizations_over_the_years_(Top10).html")

    # Top 10 organizations responsible for terrorism
    show = df.filter(f.col('gname') != 'Unknown') \
        .groupBy('gname') \
        .agg(f.sum('nkill').alias('total_nkill')) \
        .orderBy(f.desc('total_nkill')) \
        .limit(10) \
        .collect()

    x_values = list(map(lambda row: row['gname'], show))
    y_values = list(map(lambda row: row['total_nkill'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='red'
    ))
    fig.update_layout(
        xaxis_title='Terrorist Organization',
        yaxis_title='Total Killings',
        title='Top 10 organizations responsible for terrorism',
        template='plotly_dark'
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_organizations_responsible_for_terrorism.html")

    # Top 10 most brutal attack types
    show = df.groupBy('attacktype1_txt') \
        .agg(f.sum('nkill').alias('total_nkill')) \
        .orderBy(f.desc('total_nkill')) \
        .limit(10) \
        .collect()
    x_values = list(map(lambda row: row['attacktype1_txt'], show))
    y_values = list(map(lambda row: row['total_nkill'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='red'
    ))
    fig.update_layout(
        xaxis_title='Attack Type',
        yaxis_title='Total Killings',
        title='Top 10 most brutal attack types',
        template='plotly_dark'
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_most_brutal_attack_types.html")

    ### Terrorisms Abductions ###

    # Total Kidnappings over the years
    show = df.filter(df['nhostkid'] >= 0) \
        .groupBy('iyear') \
        .agg(f.sum('nhostkid').alias('total_nhostkid')) \
        .orderBy('iyear') \
        .collect()

    x_values = list(map(lambda row: row['iyear'], show))
    y_values = list(map(lambda row: row['total_nhostkid'], show))

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Total Kidnappings over the years'])
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values, mode='lines+markers', name='Kidnappings', line=dict(color='darksalmon')),
        row=1, col=1
    )
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Kidnappings', template='plotly_dark')
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Total_Kidnappings_over_the_years.html")

    # Top 10 Terrorist Organizations with Ransom Amounts
    show = df.filter(df['gname'] != "Unknown") \
        .groupBy('gname') \
        .agg(f.sum('ransom').alias('total_ransom')) \
        .orderBy(f.desc('total_ransom')) \
        .limit(10) \
        .collect()
    x_values = list(map(lambda row: row['gname'], show))
    y_values = list(map(lambda row: row['total_ransom'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='darksalmon'
    ))
    fig.update_layout(
        xaxis_title='Group',
        yaxis_title='Total Ransom Amount',
        title='Top 10 Terrorist Organizations with Ransom Amounts',
        template='plotly_dark',
        xaxis_tickangle=-45
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_Terrorist_Organizations_with_Ransom_Amounts.html")

    ### Terrorism Targets ###

    # Top 10 Target Types for Terrorists
    show = df.groupBy('targtype1_txt') \
        .agg(f.count('targtype1_txt').alias('ntargets')) \
        .orderBy(f.desc('ntargets')) \
        .limit(10) \
        .collect()

    x_values = list(map(lambda row: row['targtype1_txt'], show))
    y_values = list(map(lambda row: row['ntargets'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker=dict(color='darkred')
    ))
    fig.update_layout(
        title="Top 10 Target Types for Terrorists",
        xaxis=dict(title='Target'),
        yaxis=dict(title='Number of Events'),
        template='plotly_dark',
        xaxis_tickangle=-45,
        bargap=0.2
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_Target_Types_for_Terrorists.html")

    # Top 10 Target SubTypes for Terrorists
    show = df.groupBy('targsubtype1_txt') \
        .agg(f.count('targsubtype1_txt').alias('ntargets')) \
        .orderBy(f.desc('ntargets')) \
        .limit(10) \
        .collect()

    x_values = list(map(lambda row: row['targsubtype1_txt'], show))
    y_values = list(map(lambda row: row['ntargets'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker=dict(color='darkred')
    ))
    fig.update_layout(
        title="Top 10 Target SubTypes for Terrorists",
        xaxis=dict(title='Specific Target'),
        yaxis=dict(title='Number of Events'),
        template='plotly_dark',
        xaxis_tickangle=-45,
        bargap=0.2
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_Target_SubTypes_for_Terrorists.html")

    ### Suicide based Terrorism ###

    # Total suicide attack over the years
    show = df.groupBy('iyear') \
        .agg(f.sum('suicide').alias('total_suicide')) \
        .orderBy('iyear') \
        .collect()

    x_values = list(map(lambda row: row['iyear'], show))
    y_values = list(map(lambda row: row['total_suicide'], show))

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Total suicide attack over the years'])
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values, mode='lines+markers', name='Number of suicide attacks',
                   line=dict(color='darkmagenta')),
        row=1, col=1
    )
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Suicide Attacks', template='plotly_dark')
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Total_suicide_attack_over_the_years.html")

    ### Terrorist Nationality ###

    # Terrorist Nationality based on the number of Terrorist Attacks
    show = df.groupBy('natlty1_txt') \
        .agg(f.count('*').alias('nevents')) \
        .orderBy(f.desc('nevents')) \
        .limit(10) \
        .collect()

    x_values = list(map(lambda row: row['natlty1_txt'], show))
    y_values = list(map(lambda row: row['nevents'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='blue'
    ))

    fig.update_layout(
        title='Terrorist Nationality based on the number of Terrorist Attacks',
        xaxis_title='Terrorist Nationality',
        yaxis_title='Terrorist attacks',
        template='plotly_dark'
    )

    fig.update_xaxes(tickangle=-45)
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Terrorist_Nationality_based_on_the_number_of_Terrorist_Attacks.html")

    ### Terrorist Escape Routes ###

    # Top 10 Countries of Escape
    show = df.filter(df.kidhijcountry != "") \
        .groupBy('kidhijcountry') \
        .agg(f.count('kidhijcountry').alias('nevents')) \
        .orderBy(f.desc('nevents')) \
        .limit(10) \
        .collect()

    x_values = list(map(lambda row: row['kidhijcountry'], show))
    y_values = list(map(lambda row: row['nevents'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker=dict(color='green'),
    ))
    fig.update_layout(
        title="Top 10 Countries of Escape",
        xaxis=dict(title="Escape Location"),
        yaxis=dict(title="Terrorist attacks"),
        template='plotly_dark'
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Top10_Countries_of_Escape.html")

    ### Terrorist Claim Modes ###

    # Claim Modes by organizations
    show = df.groupBy('claimmode_txt') \
        .agg(f.count('claimmode_txt').alias('nclaims')) \
        .orderBy(f.desc('nclaims')) \
        .limit(10) \
        .collect()
    x_values = list(map(lambda row: row['claimmode_txt'], show))
    y_values = list(map(lambda row: row['nclaims'], show))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker=dict(color='darkgoldenrod')
    ))
    fig.update_layout(
        title="Claim Modes by organizations",
        xaxis=dict(title='Claim Modes'),
        yaxis=dict(title='Number of Events'),
        template='plotly_dark',
        xaxis_tickangle=-45,
        bargap=0.2
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Claim_Modes_by_organizations.html")

    ###########################
    ### Geospatial Analysis ###
    ###########################

    df = df.withColumn("date", f.to_date(f.concat(f.col("iyear").cast("string"),
                                                  f.lit("-"),
                                                  f.lpad(f.col("imonth"), 2, "0"),
                                                  f.lit("-"),
                                                  f.lpad(f.col("iday"), 2, "0")), "yyyy-MM-dd"))

    geo_data = df.filter((f.col('doubtterr') == 0) &
                         (f.col('success') == 1) &
                         (f.col('specificity') == 1) &
                         (f.col('date') >= '2010-01-01') &
                         (f.col('gname').isNotNull()) &
                         (f.col('gname') != 'Unknown') &
                         (f.col('latitude').isNotNull()) &
                         (f.col('longitude').isNotNull()) &
                         (f.col('date').isNotNull())) \
        .select("latitude", "longitude", "date", "gname")

    fig = go.Figure()
    text_data = geo_data.select("date", "gname").rdd.map(
        lambda x: f"Date: {x['date']}, Terrorist Organization: {x['gname']}" if x[
            'gname'] else f"Date: {x['date']}, Terrorist Organization: Unknown").collect()
    fig.add_trace(go.Scattergeo(
        lon=geo_data.select("longitude").rdd.flatMap(lambda x: x).collect(),
        lat=geo_data.select("latitude").rdd.flatMap(lambda x: x).collect(),
        mode='markers',
        marker=dict(color='red', size=5, opacity=0.7),
        text=text_data
    ))
    fig.update_layout(
        title="Terrorism Incidents reported from 2010-2021",
        geo=dict(
            projection=dict(type='natural earth'),
            bgcolor='black',
            landcolor='seagreen',
            oceancolor='navy',
            showland=True,
            showcountries=True,
            showocean=True,
            countrycolor='black'
        ),
        template='plotly_dark'
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Geospatial_Analysis.html")

    ########################
    ### Network Analysis ###
    ########################
    output_file_analysis.write("\n\nNetwork Analysis\n")

    # Build edges from group and region
    edges = df.filter((f.col('gname').isNotNull()) & (f.col('gname') != 'Unknown') & (f.col('region_txt').isNotNull())) \
        .select('gname', 'region_txt') \
        .distinct() \
        .rdd \
        .map(lambda r: (r[0], r[1])) \
        .collect()

    # Create NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Graph Structure
    output_file_analysis.write(f"\nNodes: {len(G.nodes)}, Edges: {len(G.edges)}")

    # Community detection
    communities = nx.algorithms.community.louvain_communities(G)
    output_file_analysis.write(f"\nNumber of communities detected: {len(communities)}")

    # Community detection using modularity-based approach
    communities_modularity = nx.algorithms.community.greedy_modularity_communities(G)
    output_file_analysis.write("\nCommunities based on Modularity Measure:\n")
    output_file_analysis.write("\n".join([str(comm) for comm in communities_modularity]))

    # Find central nodes by betweenness centrality
    centrality = nx.betweenness_centrality(G)
    output_file_analysis.write(
        f"\nTop 5 central nodes: {dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5])}")

    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    top_closeness_centrality = dict(sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])
    output_file_analysis.write(f"\nTop 5 nodes by Closeness Centrality: {top_closeness_centrality}")

    #########################
    ### Survival Analysis ###
    #########################

    df = df.withColumn("campaign_start", f.min("iyear").over(Window.partitionBy("gname")))
    df = df.withColumn("campaign_end", f.max("iyear").over(Window.partitionBy("gname")))
    df = df.withColumn("duration", f.col("campaign_end") - f.col("campaign_start"))

    show = df.filter((f.col('gname').isNotNull()) & (f.col('gname') != 'Unknown')) \
        .groupBy('gname') \
        .agg(f.mean("duration").alias("mean_duration"),
             f.min("campaign_start").alias("start_date"),
             f.max("campaign_end").alias("end_date")) \
        .orderBy(f.desc('mean_duration')) \
        .limit(50) \
        .collect()

    x_values = list(map(lambda row: row['gname'], show))
    y_values = list(map(lambda row: row['mean_duration'], show))

    hover_text = list(map(
        lambda gname, start_date, end_date: f"{gname}<br>Start Year: {start_date}<br>End Year: {end_date}",
        x_values,
        list(map(lambda row: row['start_date'], show)),
        list(map(lambda row: row['end_date'], show))
    ))

    fig = go.Figure(go.Bar(
        x=x_values,
        y=y_values,
        marker_color='red',
        hovertext=hover_text
    ))
    fig.update_layout(
        xaxis_title='Terrorist Organizations',
        yaxis_title='Duration in years',
        title='Terrorist Organizations Lifespan',
        template='plotly_dark',
        width=1000,
        height=800
    )
    fig.write_html("/Users/_ritwik_sharma_/Documents/CS_657/Final Project/Project-Files/Graphs/Survival_Analysis.html")

    #########################
    ### Anomaly Detection ###
    #########################

    # Top 10 Outlier Countries by Fatality Count
    plot_anomaly(df, 'country_txt')

    # Top 10 Outlier Cities by Fatality Count
    plot_anomaly(df.filter(f.col('city') != 'Unknown'), 'city')

    # Top 10 Outlier Terrorists Organizations by Fatality Count
    plot_anomaly(df.filter(f.col('gname') != 'Unknown'), 'gname')

    # Top 10 Outlier Terrroist Targets by Fatality Count
    plot_anomaly(df, 'targtype1_txt')

    ########################
    ### Textual Analysis ###
    ########################

    # Word Cloud: Motive for the terrorist attack
    display_wordcloud(reduce(lambda df, feature: clean_text(df, feature), ['motive'], df), 'motive')

    # Word Cloud: Ransom Note during the terrorist attack
    display_wordcloud(reduce(lambda df, feature: clean_text(df, feature), ['ransomnote'], df), 'ransomnote')

    # Word Cloud: Summary after the terrorist attack
    display_wordcloud(reduce(lambda df, feature: clean_text(df, feature), ['summary'], df), 'summary')


if __name__ == "__main__":
    main()

# Close the output file
output_file_analysis.close()

# Stop Spark session
spark.stop()

########################################################################################################
### End ###
########################################################################################################