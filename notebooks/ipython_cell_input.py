def get_candidate_specs(query_text, emb, indexer):
    nns = emb_lookup(query_text, emb, indexer, n=100)
    nn_spec_ids = [nn[0] for nn in nns]
    return nn_spec_ids

def predict_duplicates(clf, chunk_df):
    X = chunk_df.drop(['left_spec_id', 'right_spec_id'], axis=1)
    predicted_label = clf.predict(X)
    return predicted_label

def write_chunk_df(chunk_df):
    out_df = chunk_df[chunk_df.label == 1][['left_spec_id', 'right_spec_id']]
    if not out_df.empty:
        if os.path.exists(OUT_FILE_PATH):
            out_df.to_csv(OUT_FILE_PATH, mode='a', header=False, index=False)
        else:
            out_df.to_csv(OUT_FILE_PATH, index=False)

        print('Wrote ', out_df.shape[0], ' dupliates to disk')

def process_buffered_rows(row_buffer):
    chunk_df = pd.DataFrame(row_buffer, columns=columns)
    chunk_df = chunk_df[model_column_order]
    chunk_df['label'] = predict_duplicates(clf, chunk_df) 
    write_chunk_df(chumk_df)
    
row_columns = ['left_'+col for col in spec_feature_names]+['right_'+col for col in spec_feature_names]
row_buffer = []
for left_index, spec_row in tqdm(specs_df.iterrows()):
    left_spec_id = spec_row.spec_id
    candidates = get_candidate_specs(spec_row.page_title_stem, emb, indexer)
    for right_spec_id in candidates:
        if left_spec_id == right_spec_id:
            continue
        
        left_brand = specs_df.loc[left_spec_id].brand
        right_brand = specs_df.loc[right_spec_id].brand
        
        # Different brands, so we can just skip it
        if ((left_brand is not None) and (right_brand is not None) and (left_brand != right_brand)):
            continue
    
        left_row = specs_df.loc[left_spec_id][spec_feature_names]
        right_row = specs_df.loc[right_spec_id][spec_feature_names]
        row = np.concatenate([left_row.values, right_row.values])
        row_buffer.append(row)
    
    if len(row_buffer) >= buffer_size:
        process_buffered_rows(row_buffer)  
        row_buffer = []
        break

if row_buffer:
    process_buffered_rows(row_buffer)