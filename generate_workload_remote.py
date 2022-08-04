from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import operator

conf = SparkConf().setAppName("test").setMaster("spark://193.190.127.237:32467")
sc = SparkContext(conf=conf)

spark = SparkSession(sc)

ratingdf = spark.read.format("csv").option("header", "true").load("file:////home/lukas/Documents/master/masterproef/evaluation/python-script/data/ratings.csv")

# Comment out for actual test
ratingdf.describe().show()

# Calculate average rating of all movies and pick top 10 from list
ratingdf = ratingdf.withColumn("rating", ratingdf["rating"].cast(DoubleType()))

avgs = ratingdf.groupBy("movieId").avg("rating")
avgs = avgs.withColumn("rating", avgs["avg(rating)"]).select("movieId", "rating")
avgs = avgs.orderBy(avgs.rating.desc())

# Remember that Spark uses lazy evaluation. No computation is done until it encounters the next line.
# These are great for seeing if stuff works, but try to avoid them so Spark can build better overall execution plans.
avgs.limit(10).show()

# Get movies and count genres
df = spark.read.format("csv").option("header", "true").load("data/movies.csv")
genreCounts = df.select("genres").rdd.flatMap(lambda line: line[0].split("|")).map(lambda genre: (genre, 1)).reduceByKey(lambda a, b: a + b)
genreCounts.collect()

#-------------------------------------------------
# Model solution
#-------------------------------------------------
moviedf = spark.read.format("csv").option("header", "true").load("data/movies.csv")
#Join movies with the averages dataframe on movieId, but drop the duplicate column.
joinratings = moviedf.join(avgs, moviedf.movieId == avgs.movieId).drop(moviedf["movieId"]).orderBy(avgs["rating"].desc())
#Select the appropriate columns and print the first 10 items.
joinratings.select(["movieId","title","rating"]).limit(10).show()

#-------------------------------------------------
# Model solution
#-------------------------------------------------
#First, we group the ratings by userId
usergroups = ratingdf.groupBy("userId")

#Next, we calculate both rating standard deviation and number of ratings per user.
ustdev = usergroups.agg(F.stddev("rating"))
ucounts = usergroups.count()

#Now we join those calculations, rename and select the appropriate columns, and show the users we are interested in (large variety of ratings, but also a large number of ratings)
#This could be done on one line, but for the purpose of readability we do it bit by bit.
ustats = ustdev.join(ucounts, "userId").withColumn("stdevrating", ustdev["stddev_samp(rating)"]).select("userId", "stdevrating", "count")
ustats = ustats.filter(~F.isnan(ustats.stdevrating)).filter(ustats["count"] > 20).sort(ustats["stdevrating"].desc())
ustats.limit(10).show()

#-------------------------------------------------
# Model solution
#-------------------------------------------------
#First, get all the movies rated 4 or more by the chosen user
usermovies = ratingdf.filter(ratingdf.userId=="3").filter(ratingdf.rating >= 4).select("movieId")

#Get ratings for these movies that were also larger than 4, and select the user ids
otheruids = ratingdf.filter(ratingdf.rating >= 4).join(usermovies, usermovies.movieId == ratingdf.movieId).select(ratingdf.userId).distinct()

#Now use that to find the other movies rated over 4 by these users, and calculate their average rating per movie
othermovies = ratingdf.filter(ratingdf.rating >= 4).join(otheruids, otheruids.userId == ratingdf.userId) 
otheravgs = othermovies.groupBy("movieId").avg("rating")
otheravgs = otheravgs.withColumn("rating", otheravgs["avg(rating)"]).select("movieId", "rating")
otheravgs = otheravgs.orderBy(otheravgs.rating.desc())

#Finally, we merge the results with the movie names and print the first 10
withtitles = otheravgs.join(moviedf, moviedf.movieId == otheravgs.movieId).drop(otheravgs.movieId).limit(10).select("movieId", "title", "rating")
withtitles.show()

#--------------------------------------------------------------
# Model solution: taking the slower option, which has the advantage of not locking users into their own genres.
# The result shows relevant average rating (from similar users)
#--------------------------------------------------------------
#Taking the averages per movie of the previous step, join them with movie names, select the appropriate columns, and explode  The delimited "genres" column into one *row* per genre.
withtitles = otheravgs.join(moviedf, moviedf.movieId == otheravgs.movieId).drop(otheravgs.movieId).select("movieId", "genres", "title", "rating").withColumn("genres", F.explode(F.split(moviedf.genres, "\|")))
#Now, we partition the dataset by genre, ordering by rating per partition
window = Window.partitionBy(withtitles.genres).orderBy(withtitles.rating.desc())
#Finally, we add a "rank" column to clearly indicate the top 5 per genre, and show only the results with a rank from 1 to 5.
#Note that due to the splitting of genres, movies can occur in more than one genre.
withtitles = withtitles.withColumn("rank", F.row_number().over(window))
withtitles.orderBy([withtitles.genres.desc(), withtitles.rank.asc()]).filter(withtitles["rank"] < 6).show()

# Stop SparkSession
spark.stop()