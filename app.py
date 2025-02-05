from flask import Flask, jsonify, request
import difflib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import joblib
import os
from functools import wraps
from dotenv import load_dotenv
from services.data_cleaning import detect_missing_values, outlier_detection


load_dotenv()

app = Flask(__name__)

api_token = os.environ.get('API_TOKEN')

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.replace('Bearer ', '', 1)
        if not token or token != api_token:
            return jsonify({'message': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

def detect_similar_fields(fields_local, fields_new):
    """ Detects similar fields between two sets """
    field_map = {}
    for field in fields_new:
        match = difflib.get_close_matches(field, fields_local, n=1)
        if match:
            field_map[field] = match[0]  # Match found
        else:
            field_map[field] = None  # No close match found
    return field_map

@app.route('/review-mappings', methods=['POST'])
def review_field_mappings():
    local_data = request.json.get('local_data', [])
    new_data = request.json.get('new_data', [])

    if not local_data or not new_data:
        return jsonify({'error': 'Both local_data and new_data are required'}), 400

    # Get field names from both datasets
    fields_local = set(local_data[0].keys()) if local_data else set()
    fields_new = set(new_data[0].keys()) if new_data else set()

    # Automatically detect field mappings (based on identical and similar field names)
    field_map = {field: field for field in fields_local.intersection(fields_new)}
    similar_field_map = detect_similar_fields(fields_local, fields_new)

    return jsonify({
        'message': 'Review these field mappings',
        'field_mapping_suggestions': field_map,
        'similar_field_map': similar_field_map
    })


# Problem 3: Field Type Inconsistencies - Check if field types are compatible
def check_field_type_compatibility(doc1, doc2, field_map):
    """ Check if field types are compatible """
    incompatible_fields = []
    for field1, field2 in field_map.items():
        value1 = doc1.get(field1)
        value2 = doc2.get(field2)

        if value1 is not None and value2 is not None:
            if type(value1) != type(value2):
                incompatible_fields.append({
                    'field1': field1,
                    'field2': field2,
                    'type1': str(type(value1)),
                    'type2': str(type(value2))
                })
    return incompatible_fields


# Problem 4: Duplicate Document Handling - Check for duplicates by a unique identifier
def is_duplicate(doc1, doc2, key_field='_id'):
    return doc1.get(key_field) == doc2.get(key_field)


# Problem 5: Data Conflict Resolution - Resolve conflicts based on rules
def resolve_data_conflict(doc1, doc2, resolution_strategy="db1_priority"):
    merged_doc = {}
    for field in doc1.keys():
        value1 = doc1.get(field)
        value2 = doc2.get(field)
        if value1 and value2:
            if resolution_strategy == "db1_priority":
                merged_doc[field] = value1  # Prioritize local_data
            elif resolution_strategy == "db2_priority":
                merged_doc[field] = value2  # Prioritize new_data
            else:
                merged_doc[field] = value1  # Default to local_data
        else:
            merged_doc[field] = value1 or value2  # Choose whichever value exists
    return merged_doc


# Problem 6: Handling Missing Fields - Ensure all fields are present in both documents
def handle_missing_fields(doc, required_fields):
    for field in required_fields:
        if field not in doc:
            doc[field] = None  # You can use a default value here, e.g., empty string or None
    return doc


@app.route('/merge', methods=['POST'])
@require_auth
def merge_databases():
    local_data = request.json.get('local_data', [])
    new_data = request.json.get('new_data', [])
    resolution_strategy = request.args.get('resolution_strategy', 'db1_priority')

    if not new_data:
        return jsonify({'error': 'new_data is required'}), 400

    missing_new_data = detect_missing_values(new_data) # Check for missing data before the merging (Will share that later)

    # Get field names
    fields_local = set(local_data[0].keys()) if local_data else set()
    fields_new = set(new_data[0].keys()) if new_data else set()

    # Detect field mappings (assuming field names match directly)
    field_map = {field: field for field in fields_local.intersection(fields_new)}

    # Prepare for merging
    all_fields = fields_local.union(fields_new)
    merged_data = []
    nullified_fields = []  # To store the nullified fields per document

    # Keep track of processed new_data documents
    processed_new_data = set()

    # Merge existing data if any
    if local_data:
        for doc1 in local_data:
            match_index = None
            for idx, d in enumerate(new_data):
                if is_duplicate(doc1, d):
                    match_index = idx
                    break
            nullified = {}

            if match_index is not None:
                match = new_data[match_index]
                processed_new_data.add(match_index)

                # Merge data from new_data into local_data
                merged_doc = {**doc1}  # Start with doc1's data

                for field in fields_new:
                    if field not in merged_doc or not merged_doc.get(field):
                        merged_doc[field] = match.get(field)
                        # Track fields that are nullified
                        if merged_doc[field] is None:
                            nullified[field] = merged_doc[field]

                merged_doc = handle_missing_fields(merged_doc, all_fields)  # Handle missing fields
                merged_data.append(merged_doc)
            else:
                doc1 = handle_missing_fields(doc1, all_fields)  # Handle missing fields in local_data
                merged_data.append(doc1)

            nullified_fields.append(nullified)  # Add nullified info for the current document

        # Add any documents from new_data that weren't matched
        for idx, doc2 in enumerate(new_data):
            if idx not in processed_new_data:
                doc2 = handle_missing_fields(doc2, all_fields)  # Handle missing fields in new_data
                merged_data.append(doc2)

    else:
        # If local_data is empty, simply add new_data to merged_data
        for doc2 in new_data:
            doc2 = handle_missing_fields(doc2, all_fields)  # Handle missing fields in new_data
            merged_data.append(doc2)

    return jsonify({
        'message': 'Data merged successfully!',
        'merged_data': merged_data,
        'nullified_fields': nullified_fields,
        'merged_count': len(merged_data)
    })


@app.route('/ai/analyze-fields', methods=['POST'])
def ai_analyze_fields():
    """New endpoint for AI-powered field analysis"""
    try:
        data = request.get_json()
        local_data = data.get('local_data', [])
        new_data = data.get('new_data', [])

        if not local_data or not new_data:
            return jsonify({
                'error': 'Both local_data and new_data are required',
                'status': 'error'
            }), 400

        # Initialize AI components
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Analyze fields
        fields_local = set(local_data[0].keys())
        fields_new = set(new_data[0].keys())

        # Get embeddings
        local_embeddings = model.encode(list(fields_local))
        new_embeddings = model.encode(list(fields_new))

        # Calculate similarities
        similarity_matrix = cosine_similarity(new_embeddings, local_embeddings)

        field_analysis = {}
        for i, new_field in enumerate(fields_new):
            best_match_idx = np.argmax(similarity_matrix[i])
            best_match_field = list(fields_local)[best_match_idx]
            semantic_similarity = float(similarity_matrix[i][best_match_idx])

            # Calculate lexical similarity
            lexical_similarity = difflib.SequenceMatcher(
                None, new_field, best_match_field
            ).ratio()

            field_analysis[new_field] = {
                'best_match': best_match_field,
                'semantic_similarity': semantic_similarity,
                'lexical_similarity': lexical_similarity,
                'combined_score': (0.7 * semantic_similarity + 0.3 * lexical_similarity),
                'sample_values': {
                    'source': [str(d.get(new_field))[:100] for d in new_data[:3]],
                    'target': [str(d.get(best_match_field))[:100] for d in local_data[:3]]
                }
            }

        return jsonify({
            'status': 'success',
            'analysis': field_analysis,
            'statistics': {
                'total_fields_analyzed': len(fields_new),
                'potential_matches': len([f for f in field_analysis.values()
                                          if f['combined_score'] > 0.8])
            }
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/ai/review-mappings', methods=['POST'])
def ai_review_mappings():
    """AI-enhanced field mapping review"""
    try:
        data = request.get_json()
        local_data = data.get('local_data', [])
        new_data = data.get('new_data', [])
        threshold = float(request.args.get('threshold', '0.8'))

        if not local_data or not new_data:
            return jsonify({
                'error': 'Both local_data and new_data are required',
                'status': 'error'
            }), 400

        model = SentenceTransformer('all-MiniLM-L6-v2')

        fields_local = set(local_data[0].keys())
        fields_new = set(new_data[0].keys())

        # Get embeddings and similarities
        local_embeddings = model.encode(list(fields_local))
        new_embeddings = model.encode(list(fields_new))
        similarity_matrix = cosine_similarity(new_embeddings, local_embeddings)

        # Generate mapping suggestions
        mapping_suggestions = {}
        for i, new_field in enumerate(fields_new):
            scores = similarity_matrix[i]
            top_matches = [
                {
                    'field': list(fields_local)[idx],
                    'score': float(score),
                    'sample_values': [
                        str(d.get(list(fields_local)[idx]))[:100]
                        for d in local_data[:2]
                    ]
                }
                for idx, score in enumerate(scores)
                if score > threshold
            ]

            if top_matches:
                mapping_suggestions[new_field] = {
                    'matches': sorted(top_matches, key=lambda x: x['score'], reverse=True),
                    'sample_values': [str(d.get(new_field))[:100] for d in new_data[:2]]
                }

        return jsonify({
            'status': 'success',
            'mapping_suggestions': mapping_suggestions,
            'summary': {
                'fields_mapped': len(mapping_suggestions),
                'fields_unmapped': len(fields_new) - len(mapping_suggestions)
            }
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/detect-missing-values', methods=['POST'])
def detect_missing_values_endpoint():
    try:
        # Parse JSON body
        json_data = request.get_json()
        if not json_data or 'data' not in json_data:
            return jsonify({"error": "Invalid JSON or 'data' key missing"}), 400

        data = json_data['data']

        # Get optional required fields
        required_fields = json_data.get('required_fields')

        # Detect missing values
        results = detect_missing_values(data, required_fields)

        # Return results as JSON
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect-outliers', methods=['POST'])
def detect_outliers_endpoint():
    try:
        # Parse JSON body
        json_data = request.get_json()
        if not json_data or 'data' not in json_data:
            return jsonify({"error": "Invalid JSON or 'data' key missing"}), 400

        # Extract the required data
        data = json_data['data']

        # Get optional parameters
        required_fields = json_data.get('required_fields')
        contamination = json_data.get('contamination', 0.01)
        n_estimators = json_data.get('n_estimators', 100)
        random_state = json_data.get('random_state')

        # Ensure numeric types where needed
        # contamination can be float, n_estimators can be int
        try:
            contamination = float(contamination)
            n_estimators = int(n_estimators)
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

        # Detect outliers using Isolation Forest
        results = outlier_detection(
            data=data,
            required_fields=required_fields,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )

        # Return the results as JSON
        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)