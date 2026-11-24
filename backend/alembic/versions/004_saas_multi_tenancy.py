"""add saas multi tenancy, api keys, subscriptions, and audit logs

Revision ID: 004_saas_multi_tenancy
Revises: 003_injury_and_exposure_logs
Create Date: 2026-11-19 15:35:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '004_saas_multi_tenancy'
down_revision: Union[str, None] = '003_injury_and_exposure_logs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Organizations table
    op.create_table(
        'organizations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=False),
        sa.Column('subscription_tier', sa.String(length=50), nullable=False, server_default='FREE'),
        sa.Column('subscription_status', sa.String(length=50), nullable=False, server_default='ACTIVE'),
        sa.Column('is_commercial_cleared', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('research_access_permitted', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('rate_limit_per_minute', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('monthly_request_quota', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('current_month_requests', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('quota_reset_date', sa.DateTime(), nullable=False),
        sa.Column('billing_email', sa.String(length=255), nullable=True),
        sa.Column('payment_customer_id', sa.String(length=100), nullable=True),
        sa.Column('payment_subscription_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_organizations_slug'), 'organizations', ['slug'], unique=True)

    # 2. Users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=False, server_default='MEMBER'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_organization_id'), 'users', ['organization_id'], unique=False)

    # 3. API Keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('key_prefix', sa.String(length=32), nullable=False),
        sa.Column('hashed_secret', sa.String(length=128), nullable=False),
        sa.Column('scopes', sa.JSON(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('total_requests', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_keys_organization_id'), 'api_keys', ['organization_id'], unique=False)
    op.create_index(op.f('ix_api_keys_key_prefix'), 'api_keys', ['key_prefix'], unique=False)

    # 4. API Audit Logs table
    op.create_table(
        'api_audit_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('api_key_id', sa.String(length=36), nullable=True),
        sa.Column('endpoint', sa.String(length=255), nullable=False),
        sa.Column('method', sa.String(length=10), nullable=False),
        sa.Column('model_id', sa.String(length=100), nullable=True),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_audit_logs_organization_id'), 'api_audit_logs', ['organization_id'], unique=False)
    op.create_index(op.f('ix_api_audit_logs_created_at'), 'api_audit_logs', ['created_at'], unique=False)

    # 5. Subscription History table
    op.create_table(
        'subscription_history',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('previous_tier', sa.String(length=50), nullable=False),
        sa.Column('new_tier', sa.String(length=50), nullable=False),
        sa.Column('reason', sa.String(length=100), nullable=False, server_default='UPGRADE'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('changed_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_subscription_history_organization_id'), 'subscription_history', ['organization_id'], unique=False)

def downgrade() -> None:
    op.drop_table('subscription_history')
    op.drop_table('api_audit_logs')
    op.drop_table('api_keys')
    op.drop_table('users')
    op.drop_table('organizations')
