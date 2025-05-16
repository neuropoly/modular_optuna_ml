This package uses PyTest to run its tests, but due to the way its structured there is a little bit of setup required:

1. Ensure all required packages are installed (see `environment.yml` in the directory above this one)
2. Activate the environment you installed the packages into it (if installing to environment which is not your "base")
3. Add the path of Modular Optuna ML to the Python path, allowing PyTest to see its contents.

    ```bash
    cd ..
    export PYTHONPATH=$PYTHONPATH:"$PWD"
    cd tests
    ```

4. Run the tests you want. For example, to run all tests:

    ```bash
    pytest
    ```

If you are running PyTest through an IDE (such as PyCharm), you may also need to mark the `tests` directory as a testing directory to do in-place runs.
