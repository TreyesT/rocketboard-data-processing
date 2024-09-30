from flask import Flask, jsonify, request

app = Flask(__name__)

# Problem 2: Field Name Ambiguity - Allow users to review field mappings
@app.route('/review-mappings', methods=['POST'])
def review_field_mappings():
    local_data = request.json.get('local_data', [])
    new_data = request.json.get('new_data', [])

    if not local_data or not new_data:
        return jsonify({'error': 'Both local_data and new_data are required'}), 400

    # Get field names from both datasets
    fields_local = set(local_data[0].keys()) if local_data else set()
    fields_new = set(new_data[0].keys()) if new_data else set()

    # Automatically detect field mappings (based on identical field names)
    field_map = {field: field for field in fields_local.intersection(fields_new)}

    # Return field mappings for review
    return jsonify({
        'message': 'Review these field mappings',
        'field_mapping_suggestions': field_map
    })


# Problem 3: Field Type Inconsistencies - Check if field types are compatible
def check_field_type_compatibility(doc1, doc2, field_map):
    incompatible_fields = []

    for field1, field2 in field_map.items():
        value1 = doc1.get(field1, None)
        value2 = doc2.get(field2, None)

        if value1 is not None and value2 is not None:
            if type(value1) != type(value2):
                incompatible_fields.append((field1, field2, type(value1), type(value2)))

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


# Main Merge Endpoint that incorporates all problems (2-6)
@app.route('/merge', methods=['POST'])
def merge_databases():
    local_data = request.json.get('local_data', [])
    new_data = request.json.get('new_data', [])
    resolution_strategy = request.args.get('resolution_strategy', 'db1_priority')

    if not local_data or not new_data:
        return jsonify({'error': 'Both local_data and new_data are required'}), 400

    # Get field names
    fields_local = set(local_data[0].keys()) if local_data else set()
    fields_new = set(new_data[0].keys()) if new_data else set()

    # Detect field mappings (assuming field names match directly)
    field_map = {field: field for field in fields_local.intersection(fields_new)}

    # Prepare for merging
    all_fields = fields_local.union(fields_new)
    merged_data = []

    for doc1 in local_data:
        # Find matching document in new_data
        match = next((d for d in new_data if is_duplicate(doc1, d)), None)
        if match:
            # Merge data from new_data into local_data
            merged_doc = {**doc1}  # Start with doc1's data

            # Copy all fields from new_data (doc2) if they're not already in doc1
            for field in fields_new:
                if field not in merged_doc or not merged_doc.get(field):
                    merged_doc[field] = match.get(field)

            merged_doc = handle_missing_fields(merged_doc, all_fields)  # Handle missing fields
            merged_data.append(merged_doc)
        else:
            doc1 = handle_missing_fields(doc1, all_fields)  # Handle missing fields in local_data
            merged_data.append(doc1)

    # Add any documents from new_data that weren't in local_data
    for doc2 in new_data:
        if not any(is_duplicate(doc2, d) for d in local_data):
            doc2 = handle_missing_fields(doc2, all_fields)  # Handle missing fields in new_data
            merged_data.append(doc2)

    return jsonify({
        'message': 'Data merged successfully!',
        'merged_data': merged_data,
        'merged_count': len(merged_data)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

