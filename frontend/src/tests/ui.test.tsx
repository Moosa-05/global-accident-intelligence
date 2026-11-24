import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/Alert';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';

describe('UI Component Primitives', () => {
  it('renders Button with variants and sizes', () => {
    render(<Button variant="amber" size="md">Run Inference</Button>);
    const btn = screen.getByRole('button', { name: /run inference/i });
    expect(btn).toBeDefined();
    expect(btn.className).toContain('bg-amber-500');
  });

  it('renders Badge with correct styles', () => {
    render(<Badge variant="success">COMMERCIAL CLEARED</Badge>);
    const badge = screen.getByText('COMMERCIAL CLEARED');
    expect(badge).toBeDefined();
    expect(badge.className).toContain('text-emerald-300');
  });

  it('renders Alert component with title and description', () => {
    render(
      <Alert variant="warning">
        <AlertTitle>Out of Distribution</AlertTitle>
        <AlertDescription>Coordinates outside empirical training coverage.</AlertDescription>
      </Alert>
    );
    expect(screen.getByText('Out of Distribution')).toBeDefined();
    expect(screen.getByText('Coordinates outside empirical training coverage.')).toBeDefined();
  });

  it('renders Card with nested structure', () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Inference Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <span>Content Body</span>
        </CardContent>
      </Card>
    );
    expect(screen.getByText('Inference Parameters')).toBeDefined();
    expect(screen.getByText('Content Body')).toBeDefined();
  });
});
