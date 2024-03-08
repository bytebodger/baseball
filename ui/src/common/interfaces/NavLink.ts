import { ReactNode } from 'react';
import { Path } from '../enums/Path';

export interface NavLink {
   children: NavLink[],
   icon: ReactNode,
   name: string,
   page: Path,
}