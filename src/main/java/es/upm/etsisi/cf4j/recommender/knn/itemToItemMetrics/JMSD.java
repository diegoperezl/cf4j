package es.upm.etsisi.cf4j.recommender.knn.itemToItemMetrics;


import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;

/**
 * This class implements JMSD as the similarity metric for the items. The similarity metric
 * is described in: Bobadilla, J., Serradilla, F., &amp; Bernal, J. (2010). A new collaborative
 * filtering metric that improves the behavior of Recommender Systems, Knowledge-Based Systems,
 * 23 (6), 520-528.
 * 
 * @author Fernando Ortega
 */
public class JMSD extends ItemToItemMetric {

	/**
	 * Maximum difference between the ratings
	 */
	private double maxDiff;

	public JMSD(DataModel datamodel, double[][] similarities) {
		super(datamodel, similarities);
		this.maxDiff = super.datamodel.getMaxRating() - super.datamodel.getMinRating();
	}

	@Override
	public double similarity(Item item, Item otherItem) {
		int u = 0, v = 0, intersection = 0; 
		double msd = 0d;

		while (u < item.getNumberOfRatings() && v < otherItem.getNumberOfRatings()) {
			if (item.getUser(u) < otherItem.getUser(v)) {
				u++;
			} else if (item.getUser(u) > otherItem.getUser(v)) {
				v++;
			} else {
				double diff = (item.getRating(u) - otherItem.getRating(v)) / this.maxDiff;
				msd += diff * diff;
				intersection++;
				u++; 
				v++;
			}	
		}

		// If there is not ratings in common, similarity does not exists
		if (intersection == 0) return Double.NEGATIVE_INFINITY;
		
		// Return similarity
		double union = item.getNumberOfRatings() + otherItem.getNumberOfRatings() - intersection;
		double jaccard = intersection / union;
		return jaccard * (1d - (msd / intersection));
	}
}