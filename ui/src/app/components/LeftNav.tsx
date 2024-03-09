import { CalendarMonth, Home, Storage } from '@mui/icons-material';
import { ListItem, ListItemButton, ListItemText, Tooltip } from '@mui/material';
import type { MouseEvent, ReactNode } from 'react';
import { Fragment } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ShowIf } from '../../common/components/ShowIf';
import { Color } from '../../common/enums/Color';
import { Path } from '../../common/enums/Path';
import { useViewport } from '../../common/hooks/useViewport';
import type { NavLink } from '../../common/interfaces/NavLink';

export const LeftNav = () => {
   const location = useLocation();
   const navigate = useNavigate();
   const viewport = useViewport();

   const navLinks: NavLink[] = [
      {
         icon: <Home/>,
         name: 'Home',
         page: Path.home,
         children: [],
      },
      {
         icon: <Storage/>,
         name: 'Data',
         page: Path.data,
         children: [
            {
               icon: <CalendarMonth/>,
               name: 'Schedules',
               page: Path.schedules,
               children: [],
            },
         ],
      },
   ]

   const getNavLinks = (links: NavLink[], level: number): ReactNode[] => links.map(link => {
      const getChildLinks = (links: NavLink[]) => getNavLinks(links, level + 1);

      const { children, icon, name, page } = link;
      const isCurrentLink = page === location.pathname;
      const backgroundColor = isCurrentLink ? Color.greenTemplate : Color.white;
      const color = isCurrentLink ? Color.white : Color.grey;
      return (
         <Fragment key={name}>
            <ShowIf condition={!viewport.isMobile}>
               <ListItem
                  disablePadding={true}
                  key={`leftNavLink-${name}`}
                  sx={{
                     '& .MuiListItemButton-root:hover': {
                        bgcolor: isCurrentLink ? Color.greenTemplate : Color.greyLight,
                        '&, & .MuiListItemIcon-root': {
                           color,
                        },
                     },
                  }}
               >
                  <ListItemButton
                     data-value={page}
                     onClick={navigateTo}
                     sx={{
                        backgroundColor,
                        color,
                        opacity: '1 !important',
                        paddingBottom: 0,
                        paddingLeft: (level * 4) + 1,
                        paddingTop: 0,
                     }}
                  >
                     {icon}
                     <ListItemText sx={{
                        marginLeft: 1,
                        paddingTop: '4px',
                     }}>
                        {name}
                     </ListItemText>
                  </ListItemButton>
               </ListItem>
            </ShowIf>
            <ShowIf condition={viewport.isMobile}>
               <Tooltip title={name}>
                  <ListItem
                     disablePadding={true}
                     key={`leftNavLink-${name}`}
                  >
                     <ListItemButton
                        data-value={page}
                        onClick={navigateTo}
                        sx={{
                           backgroundColor,
                           color,
                           opacity: '1 !important',
                           paddingBottom: 0.9,
                           paddingLeft: (level * 2) + 1,
                           paddingTop: 0.9,
                        }}
                     >
                        {icon}
                     </ListItemButton>
                  </ListItem>
               </Tooltip>
            </ShowIf>
            {getChildLinks(children)}
         </Fragment>
      )
   })

   const navigateTo = (event: MouseEvent<HTMLElement>) => {
      const { value: navigateTo } = event.currentTarget.dataset;
      if (navigateTo) navigate(navigateTo);
   }

   return getNavLinks(navLinks, 0);
}