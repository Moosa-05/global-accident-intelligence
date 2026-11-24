"""add injury and exposure risk prediction logs

Revision ID: 003_injury_and_exposure_logs
Revises: 002_severity_and_combined_logs
Create Date: 2026-11-19 14:55:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '003_injury_and_exposure_logs'
down_revision: Union[str, None] = '002_severity_and_combined_logs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Injury Prediction Logs
    op.create_table(
        'injury_prediction_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('predicted_severity', sa.String(length=20), nullable=False),
        sa.Column('severity_level_code', sa.Integer(), nullable=False),
        sa.Column('probabilities', sa.JSON(), nullable=False),
        sa.Column('fatality_risk_alert', sa.Boolean(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('features_payload', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_injury_prediction_logs_created_at'), 'injury_prediction_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_injury_prediction_logs_latitude'), 'injury_prediction_logs', ['latitude'], unique=False)
    op.create_index(op.f('ix_injury_prediction_logs_longitude'), 'injury_prediction_logs', ['longitude'], unique=False)
    op.create_index(op.f('ix_injury_prediction_logs_predicted_severity'), 'injury_prediction_logs', ['predicted_severity'], unique=False)
    op.create_index(op.f('ix_injury_prediction_logs_fatality_risk_alert'), 'injury_prediction_logs', ['fatality_risk_alert'], unique=False)

    # 2. Exposure Risk Prediction Logs
    op.create_table(
        'exposure_risk_prediction_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('road_name', sa.String(length=100), nullable=True),
        sa.Column('road_class', sa.String(length=50), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('annual_average_daily_flow', sa.Integer(), nullable=False),
        sa.Column('link_length_km', sa.Float(), nullable=False),
        sa.Column('annual_exposure_million_vkt', sa.Float(), nullable=False),
        sa.Column('predicted_annual_crashes', sa.Float(), nullable=False),
        sa.Column('predicted_crash_rate_per_million_vkt', sa.Float(), nullable=False),
        sa.Column('risk_tier', sa.String(length=30), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_exposure_risk_prediction_logs_created_at'), 'exposure_risk_prediction_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_exposure_risk_prediction_logs_road_class'), 'exposure_risk_prediction_logs', ['road_class'], unique=False)
    op.create_index(op.f('ix_exposure_risk_prediction_logs_latitude'), 'exposure_risk_prediction_logs', ['latitude'], unique=False)
    op.create_index(op.f('ix_exposure_risk_prediction_logs_longitude'), 'exposure_risk_prediction_logs', ['longitude'], unique=False)
    op.create_index(op.f('ix_exposure_risk_prediction_logs_risk_tier'), 'exposure_risk_prediction_logs', ['risk_tier'], unique=False)

def downgrade() -> None:
    op.drop_table('exposure_risk_prediction_logs')
    op.drop_table('injury_prediction_logs')
