package es.upm.etsisi.cf4j.process;

/**
 * <p>This interface handles the parallel execution of an array of T thorough Parallelizer class. Each execution of a
 * Partible consists in three methods:</p>
 * <ol>
 *     <li>beforeRun(): is executed once before the execution of all the indices</li>
 *     <li>run(T object): is executed once per each object in the array of objects passed as parameter in the exec
 *     method of the Parallelizer class. These executions are performed in a parallel way</li>
 *     <li>afterRun(): is executed once after the execution of all the indices</li>
 * </ol>
 * @author Fernando Ortega
 */
public interface Partible<T> {

	/**
	 * Is executed once before execute the method 'exec'. It can be used to initialize resources.
	 */
	void beforeRun();

  /**
   * Is executed once for each object in the array of objects passed as parameter in the exec method of the
   * Parallelizer class. These executions are performed in a parallel way. Race conditions must be handle by
   * the developer. This method is no thread safe.
   * @param object Index of the test element.
   */
  void run(T object);

	/**
	 * Is executed once after execute the method exec. It can be used to close resources.
	 */
	void afterRun();

}
