import type { ReactNode } from 'react';

interface Props {
   children: ReactNode,
   condition: boolean,
}

export const ShowIf = ({ condition, children }: Props) => condition ? children : null;