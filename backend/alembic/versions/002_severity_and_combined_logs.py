"""add severity and combined prediction logs

Revision ID: 002_severity_and_combined_logs
Revises: 001_initial_tables
Create Date: 2026-11-19 14:10:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002_severity_and_combined_logs'
down_revision: Union[str, None] = '001_initial_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Severity Prediction Logs
    op.create_table(
        'severity_prediction_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('predicted_severity', sa.Integer(), nullable=False),
        sa.Column('severity_level', sa.String(length=30), nullable=False),
        sa.Column('probabilities', sa.JSON(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('features_payload', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_severity_prediction_logs_created_at'), 'severity_prediction_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_severity_prediction_logs_latitude'), 'severity_prediction_logs', ['latitude'], unique=False)
    op.create_index(op.f('ix_severity_prediction_logs_longitude'), 'severity_prediction_logs', ['longitude'], unique=False)
    op.create_index(op.f('ix_severity_prediction_logs_predicted_severity'), 'severity_prediction_logs', ['predicted_severity'], unique=False)

    # Combined Prediction Logs
    op.create_table(
        'combined_prediction_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('accident_alert', sa.Boolean(), nullable=False),
        sa.Column('predicted_severity', sa.Integer(), nullable=False),
        sa.Column('expected_severity_index', sa.Float(), nullable=False),
        sa.Column('joint_risk_impact_score', sa.Float(), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('payload_summary', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_combined_prediction_logs_created_at'), 'combined_prediction_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_combined_prediction_logs_risk_score'), 'combined_prediction_logs', ['risk_score'], unique=False)
    op.create_index(op.f('ix_combined_prediction_logs_predicted_severity'), 'combined_prediction_logs', ['predicted_severity'], unique=False)

def downgrade() -> None:
    op.drop_table('combined_prediction_logs')
    op.drop_table('severity_prediction_logs')
