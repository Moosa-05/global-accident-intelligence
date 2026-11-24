import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { AlertCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react';

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'info' | 'warning' | 'error' | 'success';
}

export const Alert: React.FC<AlertProps> = ({
  className,
  variant = 'info',
  children,
  ...props
}) => {
  const variants = {
    info: 'bg-sky-950/40 border-sky-800/60 text-sky-200',
    warning: 'bg-amber-950/40 border-amber-800/60 text-amber-200',
    error: 'bg-rose-950/40 border-rose-800/60 text-rose-200',
    success: 'bg-emerald-950/40 border-emerald-800/60 text-emerald-200',
  };

  const icons = {
    info: <Info className="h-4 w-4 text-sky-400 shrink-0 mt-0.5" />,
    warning: <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />,
    error: <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />,
    success: <CheckCircle className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />,
  };

  return (
    <div
      className={twMerge(
        clsx(
          'relative w-full rounded-lg border p-3 flex items-start gap-3 text-sm',
          variants[variant],
          className
        )
      )}
      role="alert"
      {...props}
    >
      {icons[variant]}
      <div className="flex-1">{children}</div>
    </div>
  );
};

export const AlertTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({ className, children, ...props }) => (
  <h5 className={twMerge(clsx('font-semibold leading-tight tracking-tight mb-1', className))} {...props}>
    {children}
  </h5>
);

export const AlertDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({ className, children, ...props }) => (
  <div className={twMerge(clsx('text-xs opacity-90', className))} {...props}>
    {children}
  </div>
);
