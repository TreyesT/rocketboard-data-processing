# Rocketboard Data Processing

This project provides a Flask-based API for processing, merging, and reviewing datasets. The API is hosted in a cloud environment using Docker containers. It focuses on solving issues related to field name ambiguities, field type inconsistencies, data duplication, and missing field handling in the context of merging two datasets.

## Overview
The API allows users to upload datasets in JSON format and performs the following tasks:

1. Field Name Ambiguity: Reviews and suggests mappings between fields with identical names in the datasets.
2. Field Type Inconsistencies: Checks whether fields of the same name have compatible types across datasets.
3. Duplicate Document Handling: Identifies duplicates based on a unique identifier (e.g., _id).
4. Data Conflict Resolution: Merges datasets based on customizable resolution strategies (e.g., prioritize one dataset over another).
5. Handling Missing Fields: Ensures all fields are present in both documents, filling in missing fields with default values (None by default).
