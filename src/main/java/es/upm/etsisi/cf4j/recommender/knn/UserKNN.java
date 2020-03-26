package es.upm.etsisi.cf4j.recommender.knn;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.utils.Parallelizer;
import es.upm.etsisi.cf4j.utils.Partible;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.knn.userSimilarityMetric.UserSimilarityMetric;
import es.upm.etsisi.cf4j.utils.Search;

/**
 * Implements user-to-user KNN based collaborative filtering
 */
public class UserKNN extends Recommender {

    /**
     * Available aggregation approaches to merge k-nearest neighbors ratings
     */
    public enum AggregationApproach {MEAN, WEIGHTED_MEAN, DEVIATION_FROM_MEAN}

    /**
     * Number of neighbors (k)
     */
    protected int numberOfNeighbors;

    /**
     * Similarity metric to compute the similarity between two users
     */
    private UserSimilarityMetric metric;

    /**
     * Aggregation approach used to aggregate k-nearest neighbors ratings
     */
    private AggregationApproach aggregationApproach;

    /**
     * Contains the neighbors indexes of each user
     */
    protected int[][] neighbors;

    /**
     * Recommender constructor
     * @param datamodel DataModel instance
     * @param numberOfNeighbors Number of neighbors (k)
     * @param metric Similarity metric to compute the similarity between two users
     * @param aggregationApproach Aggregation approach used to aggregate k-nearest neighbors ratings
     */
    public UserKNN(DataModel datamodel, int numberOfNeighbors, UserSimilarityMetric metric, AggregationApproach aggregationApproach) {
        super(datamodel);

        this.numberOfNeighbors = numberOfNeighbors;

        int numUsers = this.datamodel.getNumberOfUsers();
        this.neighbors = new int[numUsers][numberOfNeighbors];

        this.metric = metric;
        this.metric.setDatamodel(datamodel);

        this.aggregationApproach = aggregationApproach;
    }

    @Override
    public void fit() {
        Parallelizer.exec(this.datamodel.getUsers(), this.metric);
        Parallelizer.exec(this.datamodel.getUsers(), new UserNeighbors());
    }

    @Override
    public double predict(int userIndex, int itemIndex) {
        switch(this.aggregationApproach) {
            case MEAN:
                return predictMean(userIndex, itemIndex);
            case WEIGHTED_MEAN:
                return predictWeightedMean(userIndex, itemIndex);
            case DEVIATION_FROM_MEAN:
                return predictDeviationFromMean(userIndex, itemIndex);
            default:
                return Double.NaN;
        }
    }

    /**
     * Implementation of MEAN aggregation approach
     * @param userIndex user index
     * @param itemIndex item index
     * @return Ration prediction from the user to the item
     */
    private double predictMean(int userIndex, int itemIndex) {
        double prediction = 0;
        int count = 0;

        for (int neighborIndex : this.neighbors[userIndex]) {
            if (neighborIndex == -1) break; // Neighbors array are filled with -1 when no more neighbors exists

            User neighbor = this.datamodel.getUser(neighborIndex);

            int pos = neighbor.findItem(itemIndex);
            if (pos != -1) {
                prediction += neighbor.getRatingAt(pos);
                count++;
            }
        }

        if (count == 0) {
            return Double.NaN;
        } else {
            prediction /= count;
            return prediction;
        }
    }

    /**
     * Implementation of WEIGHTED_MEAN aggregation approach
     * @param userIndex user index
     * @param itemIndex item index
     * @return Ration prediction from the user to the item
     */
    private double predictWeightedMean(int userIndex, int itemIndex) {
        double[] similarities = metric.getSimilarities(userIndex);

        double num = 0;
        double den = 0;

        for (int neighborIndex : this.neighbors[userIndex]) {
            if (neighborIndex == -1) break; // Neighbors array are filled with -1 when no more neighbors exists

            User neighbor = this.datamodel.getUser(neighborIndex);

            int pos = neighbor.findItem(itemIndex);
            if (pos != -1) {
                double similarity = similarities[neighborIndex];
                double rating = neighbor.getRatingAt(pos);
                num += similarity * rating;
                den += similarity;
            }
        }

        return (den == 0) ? Double.NaN : num / den;
    }

    /**
     * Implementation of DEVIATION_FROM_MEAN aggregation approach
     * @param userIndex user index
     * @param itemIndex item index
     * @return Ration prediction from the user to the item
     */
    private double predictDeviationFromMean(int userIndex, int itemIndex) {
        User user = this.datamodel.getUser(userIndex);
        double[] similarities = metric.getSimilarities(userIndex);

        double num = 0;
        double den = 0;

        for (int neighborIndex : this.neighbors[userIndex]) {
            if (neighborIndex == -1) break; // Neighbors array are filled with -1 when no more neighbors exists

            User neighbor = this.datamodel.getUser(neighborIndex);

            int pos = neighbor.findItem(itemIndex);
            if (pos != -1) {
                double similarity = similarities[neighborIndex];
                double rating = neighbor.getRatingAt(pos);
                double avg = neighbor.getRatingAverage();

                num += similarity * (rating - avg);
                den += similarity;
            }
        }

        return (den == 0)
                ? Double.NaN
                : user.getRatingAverage() + num / den;
    }

    /**
     * Private class to parallelize neighbors computation
     */
    private class UserNeighbors implements Partible<User> {

        @Override
        public void beforeRun() { }

        @Override
        public void run(User user) {
            int userIndex = user.getUserIndex();
            double[] similarities = metric.getSimilarities(userIndex);
            neighbors[userIndex] = Search.findTopN(similarities, numberOfNeighbors);
        }

        @Override
        public void afterRun() { }
    }
}
