import { Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';
import { Loading } from '../common/components/Loading';
import { component } from '../common/constants/component';
import { Path } from '../common/enums/Path';

export const Pages = () => {
   return <>
      <Suspense fallback={<Loading open={true}/>}>
         <Routes>
            <Route
               element={component.home}
               path={Path.home}
            />
            <Route
               element={
                  <strong>
                     Page not found
                  </strong>
               }
               path={'*'}
            />
         </Routes>
      </Suspense>
   </>
}