"""initial prediction logs table

Revision ID: 001_initial_tables
Revises: 
Create Date: 2026-11-18 11:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_tables'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        'prediction_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('road', sa.String(length=50), nullable=False),
        sa.Column('mile', sa.Float(), nullable=False),
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('risk_level', sa.String(length=20), nullable=False),
        sa.Column('accident_alert', sa.Boolean(), nullable=False),
        sa.Column('threshold_used', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('features_payload', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_prediction_logs_created_at'), 'prediction_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_prediction_logs_road'), 'prediction_logs', ['road'], unique=False)
    op.create_index(op.f('ix_prediction_logs_risk_score'), 'prediction_logs', ['risk_score'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_prediction_logs_risk_score'), table_name='prediction_logs')
    op.drop_index(op.f('ix_prediction_logs_road'), table_name='prediction_logs')
    op.drop_index(op.f('ix_prediction_logs_created_at'), table_name='prediction_logs')
    op.drop_table('prediction_logs')
