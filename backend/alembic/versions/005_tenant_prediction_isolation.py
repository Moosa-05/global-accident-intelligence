"""add organization_id to prediction log tables for multi-tenant data isolation

Revision ID: 005_tenant_prediction_isolation
Revises: 004_saas_multi_tenancy
Create Date: 2026-11-19 18:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '005_tenant_prediction_isolation'
down_revision: Union[str, None] = '004_saas_multi_tenancy'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Injury Prediction Logs
    op.add_column('injury_prediction_logs', sa.Column('organization_id', sa.String(length=36), nullable=True))
    op.create_index(op.f('ix_injury_prediction_logs_organization_id'), 'injury_prediction_logs', ['organization_id'], unique=False)

    # 2. Exposure Risk Prediction Logs
    op.add_column('exposure_risk_prediction_logs', sa.Column('organization_id', sa.String(length=36), nullable=True))
    op.create_index(op.f('ix_exposure_risk_prediction_logs_organization_id'), 'exposure_risk_prediction_logs', ['organization_id'], unique=False)

    # 3. Corridor Prediction Logs
    op.add_column('prediction_logs', sa.Column('organization_id', sa.String(length=36), nullable=True))
    op.create_index(op.f('ix_prediction_logs_organization_id'), 'prediction_logs', ['organization_id'], unique=False)

    # 4. Severity Prediction Logs
    op.add_column('severity_prediction_logs', sa.Column('organization_id', sa.String(length=36), nullable=True))
    op.create_index(op.f('ix_severity_prediction_logs_organization_id'), 'severity_prediction_logs', ['organization_id'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_severity_prediction_logs_organization_id'), table_name='severity_prediction_logs')
    op.drop_column('severity_prediction_logs', 'organization_id')

    op.drop_index(op.f('ix_prediction_logs_organization_id'), table_name='prediction_logs')
    op.drop_column('prediction_logs', 'organization_id')

    op.drop_index(op.f('ix_exposure_risk_prediction_logs_organization_id'), table_name='exposure_risk_prediction_logs')
    op.drop_column('exposure_risk_prediction_logs', 'organization_id')

    op.drop_index(op.f('ix_injury_prediction_logs_organization_id'), table_name='injury_prediction_logs')
    op.drop_column('injury_prediction_logs', 'organization_id')
