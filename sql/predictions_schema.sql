CREATE TABLE IF NOT EXISTS prediction_batches (
    id uuid PRIMARY KEY,
    batch_hash text NOT NULL UNIQUE,
    model text NOT NULL,
    threshold numeric(10, 4) NOT NULL,
    total_transactions integer NOT NULL,
    created_at timestamp without time zone NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    id uuid PRIMARY KEY,
    batch_id uuid NOT NULL REFERENCES prediction_batches(id) ON DELETE CASCADE,
    trans_num text NOT NULL,
    risk_score numeric(12, 6) NOT NULL,
    prediction boolean NOT NULL,
    created_at timestamp without time zone NOT NULL DEFAULT now(),
    CONSTRAINT uq_predictions_batch_trans_num UNIQUE (batch_id, trans_num)
);

CREATE INDEX IF NOT EXISTS ix_predictions_trans_num ON predictions (trans_num);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS ix_prediction_batches_created_at ON prediction_batches (created_at);
