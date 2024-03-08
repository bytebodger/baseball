import { AppBar, Box, CssBaseline, Drawer, List, Toolbar, Typography } from '@mui/material';
import { Color } from '../../common/enums/Color';
import { HtmlElement } from '../../common/enums/HtmlElement';
import { useViewport } from '../../common/hooks/useViewport';
import { Pages } from '../../pages/Pages';
import { Header } from './Header';
import { LeftNav } from './LeftNav';

export const SiteTemplate = () => {
   const viewport = useViewport();

   const drawerWidth = viewport.isMobile ? 60 : 240;

   return <>
      <Box sx={{ display: 'flex' }}>
         <CssBaseline/>
         <AppBar
            position={'fixed'}
            sx={{
               backgroundColor: Color.greenTemplate,
               backgroundSize: '122px 55px',
               borderBottom: `6px solid ${Color.greyLight}`,
               zIndex: theme => theme.zIndex.drawer + 1,
            }}
         >
            <Toolbar>
               <Typography
                  component={HtmlElement.div}
                  noWrap={true}
                  sx={{ width: '100%' }}
                  variant={HtmlElement.h6}
               >
                  <Header/>
               </Typography>
            </Toolbar>
         </AppBar>
         <Drawer
            sx={{
               '& .MuiDrawer-paper': {
                  boxSizing: 'border-box',
                  width: drawerWidth,
               },
               flexShrink: 0,
               width: drawerWidth,
            }}
            variant={'permanent'}
         >
            <Toolbar/>
            <Box sx={{
               marginTop: 3,
               overflow: 'auto',
            }}>
               <List sx={{ padding: 0 }}>
                  <LeftNav/>
               </List>
            </Box>
         </Drawer>
         <Box
            component={HtmlElement.main}
            sx={{
               flexGrow: 1,
               p: 3,
            }}
         >
            <Toolbar/>
            <Pages/>
         </Box>
      </Box>
   </>
}