import { Box } from '@mui/material';
import { Link } from 'react-router-dom';
import { Color } from '../../common/enums/Color';
import { Path } from '../../common/enums/Path';

export const Header = () => {
   return <>
      <Box sx={{
         display: 'flex',
         justifyContent: 'space-between',
      }}>
         <Box sx={{
            marginLeft: '110px',
            marginTop: 1,
         }}>
            <Link
               style={{
                  color: Color.white,
                  fontSize: '1.4em',
                  letterSpacing: '0.1rem',
                  textDecoration: 'none',
               }}
               to={Path.home}
            >
               Baseball AI
            </Link>
         </Box>
         <Box sx={{
            alignItems: 'center',
            display: 'flex',
            textAlign: 'center',
         }}/>
      </Box>
   </>
}