#!/bin/bash

kaggle competitions download -c icr-identify-age-related-conditions
mv icr-identify-age-related-conditions.zip data/
unzip data/icr-identify-age-related-conditions.zip -d data/
