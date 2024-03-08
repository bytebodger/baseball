import { Home } from '@mui/icons-material';
import type { MouseEvent } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Path } from '../../common/enums/Path';
import { useViewport } from '../../common/hooks/useViewport';
import { NavLink } from '../../common/interfaces/NavLink';

export const LeftNav = () => {
   const location = useLocation();
   const navigate = useNavigate();
   const viewport = useViewport();

   const navLinks: NavLink[] = [
      {
         icon:
            <Home/>,
         name: 'Home',
         page: Path.home,
         children: [],
      },
   ]

   const navigateTo = (event: MouseEvent<HTMLElement>) => {
      const { value: navigateTo } = event.currentTarget.dataset;
      if (navigateTo) navigate(navigateTo);
   }

   return <></>
}