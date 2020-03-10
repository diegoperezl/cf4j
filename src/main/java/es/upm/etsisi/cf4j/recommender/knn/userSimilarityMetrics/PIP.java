package es.upm.etsisi.cf4j.recommender.knn.userSimilarityMetrics;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.data.User;

/**
 * Implements the following CF similarity metric: Ahn, H. J. (2008). A new similarity 
 * measure for collaborative filtering to alleviate the new user cold-starting problem,
 * Information Sciences, 178, 37-51.
 * 
 * @author Fernando Ortega
 */
public class PIP extends UserSimilarities {

	/**
	 * Median of the ratings of the dataset
	 */
	private double median;
	
	/**
	 * Maximum rating value
	 */
	private double max;
	
	/**
	 * Minimum rating value
	 */
	private double min;
	
	/**
	 * Constructor of the similarity metric
	 */
	public PIP (DataModel datamodel, double[][] similarities) {
		super(datamodel, similarities);

		this.max = super.datamodel.getMaxRating();
		this.min = super.datamodel.getMinRating();

		this.median = (max + min) / 2d;
	}
	
	@Override
	public double similarity(TestUser testUser, User otherUser) {

		int i = 0, j = 0, common = 0; 
		double PIP = 0d;
		
		while (i < testUser.getNumberOfRatings() && j < otherUser.getNumberOfRatings()) {
			if (testUser.getItemAt(i) < otherUser.getItemAt(j)) {
				i++;
			} else if (testUser.getItemAt(i) > otherUser.getItemAt(j)) {
				j++;
			} else {
				double ra = testUser.getRatingAt(i);
				double rt = otherUser.getRatingAt(j);
				
				// Compute agreement
				boolean agreement = true;
				if ((ra > this.median && rt < this.median) || (ra < this.median && rt > this.median)) {
					agreement = false;
				}

				// Compute proximity
				double d = (agreement) ? Math.abs(ra - rt) : 2 * Math.abs(ra - rt);
				double proximity = ((2d * (this.max - this.min) + 1d) - d) * ((2d * (this.max - this.min) + 1d) - d);

				// Compute impact
				double im = (Math.abs(ra - this.median) + 1d) * (Math.abs(rt - this.median) + 1d);
				double impact = (agreement) ? im : 1d / im;

				// Compute popularity
				String itemIndex = testUser.getItemAt(i);
				Item item = super.datamodel.getItemAt(itemIndex);
				double itemAvg = item.getRatingAverage();
				
				double popularity = 1;
				if ((ra > itemAvg && rt > itemAvg) || (ra < itemAvg && rt < itemAvg)) {
					popularity = 1d + Math.pow(((ra + rt) / 2d) - itemAvg, 2d);
				}

				// Increment PIP
				PIP += proximity * impact * popularity;
				
				common++;
				i++;
				j++;
			}	
		}

		// If there is not items in common, similarity does not exists
		if (common == 0) return Double.NEGATIVE_INFINITY;

		// Return similarity
		return PIP;
	}
}
